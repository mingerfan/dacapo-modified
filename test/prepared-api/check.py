#!/usr/bin/env python3
"""Compile opaque prepared tasks and check their RuntimePlan and placed MLIR."""

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile


# call_id -> (callee, operand call_ids, latency_us, scale_log2, physical level,
#             custom attributes that must survive placement and memory planning)
CALLS = {
    41: ("prepared_stem", [], 20, 40, 5,
         ['prepared.label = "stem"', "prepared.persistent = true"]),
    7: ("prepared_left", [41], 1000, 53, 3,
        ['prepared.label = "left"', "prepared.batch = 3 : i64"]),
    103: ("prepared_right", [41], 700, 37, 4,
          ['prepared.label = "right"', "prepared.levels = array<i64: 5, 4, 1>"]),
    23: ("prepared_join", [103, 7], 120, 48, 2,
         ['prepared.label = "join"', 'prepared.note = "operand order matters"']),
    9007199254741009: ("prepared_head", [23], 80, 45, 1,
                       ['prepared.label = "head"',
                        'prepared.extra = {coefficients = array<i64: 9, 2>, '
                        'note = "untouched"}']),
}
OUTPUT = 9007199254741009  # Above 2**53: call_id is i64, not a float/ValueId.


def poly(call_id):
    _, _, _, scale, level, _ = CALLS[call_id]
    return f"tensor<1x!ckks.poly<2 * {scale} * {level}>>"


def compact(text):
    return re.sub(r"\s+", "", text)


def integer_attr(line, name):
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(-?\d+)\s*:\s*i64", line)
    assert match, (name, line)
    return int(match[1])


def check_mlir(text, device_count):
    lines = [line for line in text.splitlines() if '"ckks.apicall"' in line]
    assert len(lines) == len(CALLS), text
    placed = {}
    for line in lines:
        call_id = integer_attr(line, "call_id")
        assert call_id in CALLS and call_id not in placed, line
        callee, inputs, latency, _, _, extra = CALLS[call_id]
        assert f'callee = "{callee}"' in line, line
        assert integer_attr(line, "latency_us") == latency, line
        for attribute in extra:
            assert compact(attribute) in compact(line), (attribute, line)
        signature = re.search(r":\s*\(([^)]*)\)\s*->\s*(.+)$", line)
        assert signature, line
        assert compact(signature[1]) == compact(", ".join(poly(i) for i in inputs)), line
        assert compact(signature[2]) == compact(poly(call_id)), line
        placed[call_id] = {
            "rank": integer_attr(line, "dist.rank"),
            "device": integer_attr(line, "dist.device"),
            "start": integer_attr(line, "dist.schedule_start"),
            "finish": integer_attr(line, "dist.schedule_finish"),
        }
    assert set(placed) == set(CALLS)
    for call_id, interval in placed.items():
        assert interval["rank"] == 0
        assert 0 <= interval["device"] < device_count
        assert 0 <= interval["start"] < interval["finish"]
        assert interval["finish"] - interval["start"] == CALLS[call_id][2]
        for input_id in CALLS[call_id][1]:
            producer = placed[input_id]
            communication = int(producer["device"] != interval["device"])
            assert interval["start"] >= producer["finish"] + communication
    for device in range(device_count):
        intervals = sorted((i["start"], i["finish"]) for i in placed.values()
                           if i["device"] == device)
        assert all(left[1] <= right[0] for left, right in zip(intervals, intervals[1:]))
    assert {i["device"] for i in placed.values()} == set(range(device_count))
    assert text.count('"ckks.') == len(CALLS), text
    assert "runtime.memory_planned" in text
    assert "dist.release" in text
    assert ("dist.transfer" in text) == (device_count == 2)
    assert "runtime.reuse_input" not in text
    return placed


def check_plan(plan, placed, device_count):
    assert plan["format_version"] == 7
    assert plan["target"]["world_size"] == 1
    assert plan["target"]["device_counts"] == [device_count]
    assert plan["external_inputs"] == []
    assert plan["initialization"] == []
    assert plan["finalization"] == []
    values = {value["id"]: value for value in plan["values"]}
    assert len(values) == len(plan["values"])
    assert all(isinstance(i, str) and str(int(i)) == i for i in values)
    available, origin, seen = set(), {}, set()
    released = set()
    for ordinal, instruction in enumerate(plan["execution"]):
        assert instruction["ordinal"] == ordinal
        kind = instruction["kind"]
        if kind == "release":
            value = instruction["value"]
            assert value in available and value not in plan["final_outputs"]
            available.remove(value)
            released.add(value)
            continue
        inputs = instruction["inputs"]
        assert all(i in available for i in inputs), instruction
        if kind == "transfer":
            assert len(inputs) == len(instruction["outputs"]) == 1
            output = instruction["outputs"][0]
            assert output not in origin
            origin[output] = origin[inputs[0]]
            source_value, output_value = values[inputs[0]], values[output]
            assert instruction["sources"] == [source_value["place"]]
            assert instruction["destinations"] == [output_value["place"]]
            metadata = lambda value: {k: v for k, v in value.items() if k not in ("id", "place")}
            assert metadata(source_value) == metadata(output_value)
        else:
            assert kind == "compute" and instruction["op"] == "ApiCall", instruction
            attrs = instruction["attrs"]
            assert isinstance(attrs["call_id"], str)
            call_id = int(attrs["call_id"])
            assert str(call_id) == attrs["call_id"]
            assert call_id in CALLS and call_id not in seen
            callee, input_ids, _, scale, level, _ = CALLS[call_id]
            assert attrs == {"callee": callee, "call_id": str(call_id)}
            assert [origin[i] for i in inputs] == input_ids, instruction
            assert all(i in seen for i in input_ids), instruction
            output = instruction["output"]
            assert output not in origin
            origin[output] = call_id
            seen.add(call_id)
            place = {"kind": "device", "rank": 0, "index": placed[call_id]["device"]}
            assert instruction["place"] == place
            assert "reuse_input" not in instruction
            value = values[output]
            assert value["kind"] == "ciphertext" and value["components"] == 2
            assert value["level"] == level and value["scale_log2"] == scale
            assert value["place"] == place and value["context"] == "test-context"
        assert output in values
        available.add(output)
    assert seen == set(CALLS)
    assert set(values) == set(origin)
    assert len(plan["final_outputs"]) == 1
    assert [origin[i] for i in plan["final_outputs"]] == [OUTPUT]
    assert available == set(plan["final_outputs"])
    assert released


class BinaryPlan:
    """Read just the V7 record kinds used by this fixture; reject all others."""

    def __init__(self, data):
        self.data = data
        self.cursor = 0
        self.strings = []

    def take(self, format_):
        format_ = "<" + format_
        value, = struct.unpack_from(format_, self.data, self.cursor)
        self.cursor += struct.calcsize(format_)
        return value

    def text(self):
        size = self.take("I")
        end = self.cursor + size
        assert end <= len(self.data)
        value = self.data[self.cursor:end].decode("utf-8")
        self.cursor = end
        return value

    def sequence(self, read):
        return [read() for _ in range(self.take("Q"))]

    def id(self):
        return str(self.take("Q"))

    def string_ref(self):
        return self.strings[self.take("I")]

    def place(self):
        kind, rank, index = self.take("B"), self.take("i"), self.take("i")
        assert kind == 1  # This fixture has only device values.
        return {"kind": "device", "rank": rank, "index": index}

    def value(self):
        result = {"id": self.id()}
        assert self.take("B") == 1  # ciphertext
        result.update(kind="ciphertext", place=self.place(), context=self.string_ref(),
                      level=self.take("i"), scale_log2=self.take("d"))
        ntt = self.take("B")
        assert ntt in (0, 1)
        result.update(ntt=bool(ntt), components=self.take("i"))
        return result

    def instruction(self):
        result = {"ordinal": self.take("Q")}
        tag = self.take("B")
        if tag == 1:
            assert self.take("B") == 13  # ApiCall; opcode 12 is reserved for V6.
            result.update(kind="compute", op="ApiCall", inputs=self.sequence(self.id),
                          output=self.id(), place=self.place())
            assert self.take("B") == 6  # ApiCall attrs; tag 5 is reserved for V6.
            result["attrs"] = {"callee": self.string_ref(), "call_id": self.id()}
            assert self.take("B") == 0  # No reuse_input for opaque callbacks.
        elif tag == 2:
            result.update(kind="transfer", transfer_id=self.id())
            assert self.take("B") == 0  # No collective group.
            assert self.take("B") == 1  # point_to_point
            result.update(hint="point_to_point", inputs=self.sequence(self.id),
                          outputs=self.sequence(self.id), sources=self.sequence(self.place),
                          destinations=self.sequence(self.place))
            kinds = self.sequence(lambda: self.take("B"))
            assert kinds == [1]
            result["output_kinds"] = ["ciphertext"]
        else:
            assert tag == 3, tag
            result.update(kind="release", value=self.id())
        return result

    def read(self):
        assert self.data[:8] == b"CKKSPL01"
        self.cursor = 8
        assert self.take("I") == 1  # Binary container version, not semantic V7.
        plan = json.loads(self.text())
        assert plan["format_version"] == 7
        self.strings = self.sequence(self.text)
        for section, read in (("values", self.value), ("external_inputs", self.id),
                              ("initialization", self.instruction),
                              ("execution", self.instruction),
                              ("finalization", self.instruction), ("final_outputs", self.id)):
            assert plan[section] == []
            plan[section] = self.sequence(read)
        assert self.cursor == len(self.data)
        return plan


def invalid_call(callee='"prepared"', call_id=1, latency=1, components=2):
    type_ = f"tensor<1x!ckks.poly<{components} * 40 * 5>>"
    return (f"module {{ func.func @invalid() -> {type_} {{\n"
            f'%x = "ckks.apicall"() {{callee = {callee}, call_id = {call_id} : i64, '
            f"latency_us = {latency} : i64}} : () -> {type_}\n"
            f"return %x : {type_}\n}} }}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    binary = args.hecate_opt.resolve()
    fixture = source / "test/prepared-api/branching.mlir"
    with tempfile.TemporaryDirectory(prefix="dacapo-prepared-api-") as directory:
        temp = Path(directory)

        def run(path, output, options=(), expected=()):
            command = [str(binary), str(path), f"--ckks-config={source / 'config.json'}",
                       *options, "-o", str(output)]
            result = subprocess.run(command, cwd=source, capture_output=True,
                                    text=True, timeout=60)
            if expected:
                assert result.returncode != 0, f"unexpected success: {command}"
                assert all(word in result.stderr for word in expected), result.stderr
            else:
                assert result.returncode == 0, f"{command}\n{result.stderr}"
            return result

        common = [
            "--ckks-to-runtime-plan", "--runtime-plan-coalesce-rescales=false",
            "--runtime-plan-levels-per-logical-level=1",
            f"--runtime-plan-operator-spec-path={source / 'test/runtime-plan/placement-operator-spec.json'}",
            "--runtime-plan-intra-rank-communication-cost=1",
            "--runtime-plan-inter-rank-communication-cost=1",
            "--runtime-plan-memory", "--runtime-plan-memory-reuse=false",
            "--runtime-plan-target-id=test", "--runtime-plan-operator-spec-id=test",
            f"--runtime-plan-operator-spec-sha256=sha256:{'0' * 64}",
            "--runtime-plan-context-id=test-context",
        ]
        for device_count in (1, 2):
            plans = []
            for format_ in ("json", "binary"):
                prefix = temp / f"gpu{device_count}-{format_}"
                output = prefix.with_suffix(".mlir")
                run(fixture, output, [*common,
                                     f"--runtime-plan-device-counts={device_count}",
                                     f"--runtime-plan-format={format_}"])
                placed = check_mlir(output.read_text(), device_count)
                extension = "json" if format_ == "json" else "bin"
                path = Path(f"{prefix}.prepared_branching.runtime-plan.{extension}")
                plan = (json.loads(path.read_text()) if format_ == "json"
                        else BinaryPlan(path.read_bytes()).read())
                check_plan(plan, placed, device_count)
                plans.append(plan)
            assert plans[0] == plans[1], "JSON and binary RuntimePlans differ"

        duplicate = temp / "duplicate.mlir"
        duplicate.write_text(fixture.read_text().replace("call_id = 7 : i64",
                                                        "call_id = 41 : i64"))
        run(duplicate, temp / "duplicate.out.mlir",
            [*common, "--runtime-plan-device-counts=1", "--runtime-plan-format=json"],
            expected=("duplicate", "call_id"))

        reuse = temp / "reuse.mlir"
        reuse.write_text(invalid_call().replace(
            "latency_us = 1 : i64", "latency_us = 1 : i64, runtime.reuse_input = 0 : i64"))
        run(reuse, temp / "reuse.out.mlir", [
            f"--emit-runtime-plan=prefix={temp / 'reuse'} plan-format=json "
            "target-id=test operator-spec-id=test "
            f"operator-spec-sha256=sha256:{'0' * 64} context-id=test-context"],
            expected=("prepared API calls", "runtime.reuse_input"))

        for name, content, diagnostic in (
            ("empty-callee", invalid_call(callee='""'), ("callee", "nonempty")),
            ("negative-call-id", invalid_call(call_id=-1), ("call_id", "nonnegative")),
            ("zero-latency", invalid_call(latency=0), ("latency_us", "positive")),
            ("negative-latency", invalid_call(latency=-1), ("latency_us", "positive")),
            ("plaintext-result", invalid_call(components=1), ("result", "ciphertext")),
        ):
            path = temp / f"{name}.mlir"
            path.write_text(content)
            run(path, temp / f"{name}.out.mlir", expected=diagnostic)
    print("Prepared API placement, RuntimePlan and verifier checks passed")


if __name__ == "__main__":
    main()
