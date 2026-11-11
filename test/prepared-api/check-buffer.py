#!/usr/bin/env python3
"""Check prepared hoist buffers, RuntimePlan V8, lifetimes and byte accounting."""

import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile


CT = "tensor<1x!ckks.poly<2 * 40 * 5>>"
PT = "tensor<1x!ckks.poly<1 * 40 * 5>>"
BUFFER = '!ckks.buffer<1234, "ckks.hoisted.v1">'
PLACE = {"kind": "device", "rank": 0, "index": 0}
# call_id -> (callee, input producers, latency_us)
CALLS = {
    10: ("prepared_stem", [], 20),
    20: ("prepared_hoist", [10], 100),
    30: ("prepared_rotate_left", [10, 20], 70),
    40: ("prepared_rotate_right", [10, 20], 80),
    60: ("prepared_boot_q30", ["sum"], 500),
}


def integer_attr(line, name):
    match = re.search(rf"\b{re.escape(name)}\s*=\s*(-?\d+)\s*:\s*i64", line)
    assert match, (name, line)
    return int(match[1])


def check_mlir(text):
    assert "runtime.memory_planned" in text
    assert "dist.transfer" not in text
    assert "ckks.bootstrapc" not in text  # Boot remains an opaque callback.
    assert "runtime.reuse_input" not in text
    assert text.count('"ckks.apicall"') == len(CALLS)
    assert text.count('"ckks.addcc"') == 1
    assert len(re.findall(r'"ckks\.\w+"\(', text)) == len(CALLS) + 1
    placed = {}
    for line in text.splitlines():
        if '"ckks.apicall"' in line:
            call_id = integer_attr(line, "call_id")
            assert call_id in CALLS and call_id not in placed, line
            callee, _, latency = CALLS[call_id]
            assert f'callee = "{callee}"' in line, line
            assert integer_attr(line, "latency_us") == latency, line
            if call_id == 20:
                assert 'prepared.label = "shared hoist"' in line
                assert re.search(r'->\s*!ckks.buffer<1234,\s*"ckks.hoisted.v1">', line)
            if call_id in (30, 40):
                assert integer_attr(line, "prepared.steps") == (1 if call_id == 30 else -1)
            if call_id == 60:
                scale = re.search(r"runtime.scale_log2\s*=\s*([\d.eE+-]+)\s*:\s*f64", line)
                assert scale and float(scale[1]) == 40.125, line
                assert re.search(r"->\s*tensor<1x!ckks.poly<2\s*\*\s*40\s*\*\s*29>>", line)
            key = call_id
        elif '"ckks.addcc"' in line:
            key, latency = "sum", None
        else:
            continue
        assert integer_attr(line, "dist.rank") == 0
        assert integer_attr(line, "dist.device") == 0
        start = integer_attr(line, "dist.schedule_start")
        finish = integer_attr(line, "dist.schedule_finish")
        assert 0 <= start < finish
        if latency is not None:
            assert finish - start == latency
        placed[key] = (start, finish)
    assert set(placed) == {*CALLS, "sum"}
    for key, (_, inputs, _) in CALLS.items():
        assert all(placed[key][0] >= placed[i][1] for i in inputs)
    assert all(placed["sum"][0] >= placed[i][1] for i in (30, 40))
    intervals = sorted(placed.values())
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    return placed


def check_plan(plan, context):
    assert plan["format_version"] == 8
    assert plan["target"]["world_size"] == 1
    assert plan["target"]["device_counts"] == [1]
    assert plan["external_inputs"] == []
    assert plan["initialization"] == plan["finalization"] == []
    values = {v["id"]: v for v in plan["values"]}
    assert len(values) == len(plan["values"]) == 6
    for id_, value in values.items():
        assert isinstance(id_, str) and str(int(id_)) == id_
        assert value["place"] == PLACE and value["context"] == context
        if value["kind"] != "buffer":
            assert value["kind"] == "ciphertext" and value["components"] == 2
            assert "buffer_bytes" not in value and "buffer_layout" not in value
    live, origins, outputs, seen = set(), {}, {}, set()
    releases, positions = {}, {}
    for ordinal, op in enumerate(plan["execution"]):
        assert op["ordinal"] == ordinal
        if op["kind"] == "release":
            id_ = op["value"]
            assert id_ in live and id_ not in plan["final_outputs"], op
            live.remove(id_)
            releases[id_] = ordinal
            continue
        assert op["kind"] == "compute" and op["place"] == PLACE, op
        assert all(i in live for i in op["inputs"]), op
        assert "reuse_input" not in op  # Also run with reuse enabled.
        if op["op"] == "ApiCall":
            attrs = op["attrs"]
            call_id = int(attrs["call_id"])
            assert attrs["call_id"] == str(call_id)
            assert call_id in CALLS and call_id not in seen
            callee, inputs, _ = CALLS[call_id]
            assert attrs == {"callee": callee, "call_id": str(call_id)}
            assert [origins[i] for i in op["inputs"]] == inputs, op
            key = call_id
            seen.add(call_id)
        else:
            assert op["op"] == "add_cc" and "sum" not in outputs, op
            assert [origins[i] for i in op["inputs"]] == [30, 40], op
            key = "sum"
        output = op["output"]
        assert output in values and output not in origins
        origins[output], outputs[key], positions[key] = key, output, ordinal
        live.add(output)
    assert seen == set(CALLS) and set(origins) == set(values)
    assert plan["final_outputs"] == [outputs[60]]
    assert live == {outputs[60]}
    assert set(releases) == set(values) - live
    hoisted = values[outputs[20]]
    assert hoisted == {
        "id": outputs[20], "kind": "buffer", "place": PLACE, "context": context,
        "level": 0, "scale_log2": 0, "ntt": False, "components": 0,
        "buffer_bytes": "1234", "buffer_layout": "ckks.hoisted.v1",
    }
    # Both suffixes consumed the same H, with no copies or early Release.
    assert releases[outputs[20]] > max(positions[30], positions[40])
    assert releases[outputs[20]] < positions["sum"]
    for key in (10, 30, 40, "sum", 60):
        value = values[outputs[key]]
        assert value["level"] == (29 if key == 60 else 5)
        assert value["scale_log2"] == 40.125
    return outputs


def check_memory(before, after, outputs, degree):
    assert before["poly_degree"] == after["poly_degree"] == degree
    assert before["memory_planned"] is False and after["memory_planned"] is True
    ct = degree * 6 * 2 * 4
    boot = degree * 30 * 2 * 4
    total = 4 * ct + boot + 1234
    for report in (before, after):
        assert report["word_bytes"] == {"host": 8, "device": 4}
        rows = {json.dumps(r["place"], sort_keys=True): r for r in report["places"]}
        assert len(rows) == 2  # Host plus the single device.
        host = rows[json.dumps({"kind": "host", "rank": 0}, sort_keys=True)]
        assert host["peak_bytes"] == host["distinct_allocation_bytes"] == host["retained_bytes"] == 0
        gpu = rows[json.dumps(PLACE, sort_keys=True)]
        assert gpu["distinct_allocation_bytes"] == total
        assert gpu["peak"]["plaintext_bytes"] == 0
        if report is before:
            assert gpu["peak_bytes"] == gpu["retained_bytes"] == total
            assert gpu["peak"]["ciphertext_bytes"] == 4 * ct + boot
            assert gpu["peak"]["buffer_bytes"] == 1234
            buffer = next(o for o in gpu["peak"]["largest_objects"]
                          if o["value_id"] == outputs[20])
            assert buffer == {"value_id": outputs[20], "kind": "buffer",
                              "bytes": 1234, "external": False}
        else:
            assert gpu["peak_bytes"] == ct + boot
            assert gpu["retained_bytes"] == boot
            assert gpu["peak"]["ciphertext_bytes"] == ct + boot
            assert gpu["peak"]["buffer_bytes"] == 0


class BinaryPlan:
    """Decode only the V8 records used here, independently of the exporter."""

    def __init__(self, data):
        self.data, self.cursor, self.strings = data, 0, []

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
        assert kind in (0, 1)
        result = {"kind": "device" if kind else "host", "rank": rank}
        if kind:
            result["index"] = index
        else:
            assert index == 0
        return result

    def value(self):
        start = self.cursor
        result = {"id": self.id()}
        kind = self.take("B")
        assert kind in (0, 1, 2)
        result.update(kind=("plaintext", "ciphertext", "buffer")[kind],
                      place=self.place(), context=self.string_ref(),
                      level=self.take("i"), scale_log2=self.take("d"))
        ntt = self.take("B")
        assert ntt in (0, 1)
        result.update(ntt=bool(ntt), components=self.take("i"))
        assert self.cursor - start == 39
        size, layout = self.id(), self.string_ref()
        assert self.cursor - start == 51  # Fixed tail for EVERY V8 value.
        if kind == 2:
            result.update(buffer_bytes=size, buffer_layout=layout)
        else:
            assert size == "0" and layout == ""
        return result

    def instruction(self):
        result = {"ordinal": self.take("Q")}
        tag = self.take("B")
        if tag == 0:
            result.update(kind="encode", output=self.id())
            assert self.take("B") == 0  # Inline plaintext payload.
            result["payload"] = {"kind": "inline", "values": self.sequence(lambda: self.take("d"))}
        elif tag == 1:
            opcode = self.take("B")
            assert opcode in (0, 1, 13)  # add_cc, add_cp or ApiCall; never reserved 12.
            result.update(kind="compute", op={0: "add_cc", 1: "add_cp", 13: "ApiCall"}[opcode],
                          inputs=self.sequence(self.id), output=self.id(), place=self.place())
            attrs = self.take("B")
            if opcode == 13:
                assert attrs == 6  # Never reserved attribute tag 5.
                start = self.cursor
                result["attrs"] = {"callee": self.string_ref(), "call_id": self.id()}
                assert self.cursor - start == 12
            else:
                assert attrs == 0
            assert self.take("B") == 0  # Opaque tasks cannot reuse input storage.
        else:
            assert tag == 3, tag  # No implicit Buffer transfers.
            result.update(kind="release", value=self.id())
        return result

    def read(self):
        assert self.data[:8] == b"CKKSPL01"
        self.cursor = 8
        assert self.take("I") == 1  # Container version, not semantic V8.
        plan = json.loads(self.text())
        assert plan["format_version"] == 8
        self.strings = self.sequence(self.text)
        for section, read in (("values", self.value), ("external_inputs", self.id),
                              ("initialization", self.instruction),
                              ("execution", self.instruction),
                              ("finalization", self.instruction), ("final_outputs", self.id)):
            assert plan[section] == []
            plan[section] = self.sequence(read)
        assert self.cursor == len(self.data)
        return plan


def call(inputs="", input_types="", result=CT, attrs=""):
    return (f'"ckks.apicall"({inputs}) {{callee = "prepared", call_id = 1 : i64, '
            f'latency_us = 1 : i64{attrs}}} : ({input_types}) -> {result}')


def module(body, result=CT, arguments="", attributes=""):
    return (f"module {{ func.func @test({arguments}) -> {result} {attributes} {{\n"
            f"{body}\n}} }}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hecate-opt", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    args = parser.parse_args()
    source, binary = args.source_dir.resolve(), args.hecate_opt.resolve()
    fixture = source / "test/prepared-api/hoist-buffer.mlir"
    spec_path = source / "test/runtime-plan/placement-operator-spec.json"
    spec = json.loads(spec_path.read_text())
    context, degree = spec["context"]["context_id"], spec["context"]["poly_degree"]
    assert len(spec["operators"]["add_cc"]["latency_us_by_level"]) >= 6
    with tempfile.TemporaryDirectory(prefix="dacapo-hoist-buffer-") as directory:
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

        def emit(prefix, format_):
            return (f"emit-runtime-plan{{prefix={prefix} plan-format={format_} "
                    f"target-id=test operator-spec-id=test context-id={context} "
                    f"operator-spec-sha256=sha256:{'0' * 64}}}")

        def load(prefix, function, format_):
            path = Path(f"{prefix}.{function}.runtime-plan.{'json' if format_ == 'json' else 'bin'}")
            return (json.loads(path.read_text()) if format_ == "json"
                    else BinaryPlan(path.read_bytes()).read())

        common = [
            "--ckks-to-runtime-plan", "--runtime-plan-coalesce-rescales=false",
            "--runtime-plan-levels-per-logical-level=1",
            f"--runtime-plan-operator-spec-path={spec_path}",
            "--runtime-plan-device-counts=1", "--runtime-plan-memory",
            "--runtime-plan-memory-report", "--runtime-plan-target-id=test",
            "--runtime-plan-operator-spec-id=test",
            f"--runtime-plan-operator-spec-sha256=sha256:{'0' * 64}",
            f"--runtime-plan-context-id={context}",
        ]
        plans = []
        for reuse in (False, True):
            for format_ in ("json", "binary"):
                prefix = temp / f"reuse{int(reuse)}-{format_}"
                output = prefix.with_suffix(".mlir")
                run(fixture, output, [*common, f"--runtime-plan-memory-reuse={str(reuse).lower()}",
                                     f"--runtime-plan-format={format_}"])
                check_mlir(output.read_text())
                plan = load(prefix, "prepared_hoist_buffer", format_)
                outputs = check_plan(plan, context)
                before = json.loads(Path(f"{prefix}.prepared_hoist_buffer.memory.before.json").read_text())
                after = json.loads(Path(f"{prefix}.prepared_hoist_buffer.memory.json").read_text())
                check_memory(before, after, outputs, degree)
                plans.append(plan)
        assert all(plan == plans[0] for plan in plans), "JSON/binary or reuse modes disagree"

        # Also cover zero-byte buffers, the plaintext V8 tail, raw three-component
        # results, and real (fractional) scales of external and prepared values.
        empty = '!ckks.buffer<0, "ckks.empty.v1">'
        raw = "tensor<1x!ckks.poly<3 * 80 * 5>>"
        metadata = temp / "metadata.mlir"
        metadata.write_text(module(
            f'%pt = "ckks.encode"() {{payload = dense<1.0> : tensor<1xf64>, '
            f'runtime.scale_log2 = 40.25 : f64}} : () -> {PT}\n'
            f'%h = {call("%arg0", CT, empty)}\n'
            f'%adjusted = "ckks.addcp"(%arg0, %pt) : ({CT}, {PT}) -> {CT}\n'
            f'%raw = "ckks.apicall"(%adjusted, %h) {{callee = "raw_mul", call_id = 2 : i64, '
            f'latency_us = 1 : i64, runtime.scale_log2 = 80.125 : f64}} : ({CT}, {empty}) -> {raw}\n'
            f'return %raw : {raw}', result=raw, arguments=f"%arg0: {CT}",
            attributes="attributes {earth.arg_scales = array<f64: 40.25>}"))
        metadata_plans = []
        for format_ in ("json", "binary"):
            prefix = temp / f"metadata-{format_}"
            run(metadata, prefix.with_suffix(".mlir"), [
                f"-p=builtin.module(func.func(plan-runtime-memory,{emit(prefix, format_)}))"])
            plan = load(prefix, "test", format_)
            values = {v["id"]: v for v in plan["values"]}
            assert len(plan["external_inputs"]) == len(plan["final_outputs"]) == 1
            assert values[plan["external_inputs"][0]]["scale_log2"] == 40.25
            result = values[plan["final_outputs"][0]]
            assert result["components"] == 3 and result["scale_log2"] == 80.125
            buffers = [v for v in values.values() if v["kind"] == "buffer"]
            assert len(buffers) == 1
            assert buffers[0]["buffer_bytes"] == "0"
            assert buffers[0]["buffer_layout"] == "ckks.empty.v1"
            assert any(v["kind"] == "plaintext" for v in values.values())
            assert all("buffer_bytes" not in v and "buffer_layout" not in v
                       for v in values.values() if v["kind"] != "buffer")
            assert all(v["context"] == context for v in values.values())
            assert not any("reuse_input" in op for op in plan["execution"])
            assert any(op["kind"] == "release" and op["value"] == buffers[0]["id"]
                       for op in plan["execution"])
            metadata_plans.append(plan)
        assert metadata_plans[0] == metadata_plans[1]

        # These inputs really reside on the GPU before execution. Placement
        # must start there, not insert and later suppress a Host upload.
        for count in (2, 27):
            arguments = ", ".join(f"%arg{i}: {CT}" for i in range(count))
            inputs = ", ".join(f"%arg{i}" for i in range(count))
            types = ", ".join(CT for _ in range(count))
            scales = ", ".join("40.125" for _ in range(count))
            header = "func.func @prepared_hoist_buffer() -> tensor<1x!ckks.poly<2 * 40 * 29>> {"
            resident = fixture.read_text().replace(header,
                f"func.func @prepared_hoist_buffer({arguments}) -> tensor<1x!ckks.poly<2 * 40 * 29>> "
                f"attributes {{runtime.external_inputs_device = 0 : i64, "
                f"earth.arg_scales = array<f64: {scales}>}} {{")
            resident = resident.replace('"ckks.apicall"()', f'"ckks.apicall"({inputs})', 1)
            resident = resident.replace(": () ->", f": ({types}) ->", 1)
            path = temp / f"resident-{count}.mlir"
            path.write_text(resident)
            resident_plans = []
            for format_ in ("json", "binary"):
                prefix = temp / f"resident-{count}-{format_}"
                output = prefix.with_suffix(".mlir")
                run(path, output, [*common, "--runtime-plan-memory-reuse=true",
                                   f"--runtime-plan-format={format_}"])
                assert "dist.transfer" not in output.read_text()
                assert 'dist.schedule_start = 0 : i64' in output.read_text()
                plan = load(prefix, "prepared_hoist_buffer", format_)
                assert plan["format_version"] == 8
                assert len(plan["external_inputs"]) == count
                assert plan["initialization"] == []
                assert all(op["kind"] != "transfer" and "reuse_input" not in op
                           for op in plan["execution"])
                values = {v["id"]: v for v in plan["values"]}
                for id_ in plan["external_inputs"]:
                    value = values[id_]
                    assert value["kind"] == "ciphertext" and value["place"] == PLACE
                    assert value["scale_log2"] == 40.125 and value["context"] == context
                first = plan["execution"][0]
                assert first["kind"] == "compute" and first["inputs"] == plan["external_inputs"]
                memory = json.loads(Path(f"{prefix}.prepared_hoist_buffer.memory.json").read_text())
                for row in memory["places"]:
                    if row["place"]["kind"] == "host":
                        assert row["peak_bytes"] == row["retained_bytes"] == 0
                    else:
                        assert row["retained_bytes"] == count * degree * 6 * 2 * 4 + degree * 30 * 2 * 4
                resident_plans.append(plan)
            assert resident_plans[0] == resident_plans[1]

        # Device-resident external CT is itself a V8 feature, even when no
        # buffers or ApiCalls are present. Every binary CT still has a zero tail.
        direct = temp / "resident-add.mlir"
        direct.write_text(module(
            f'%sum = "ckks.addcc"(%a, %b) : ({CT}, {CT}) -> {CT}\nreturn %sum : {CT}',
            arguments=f"%a: {CT}, %b: {CT}", attributes=
            "attributes {runtime.external_inputs_device = 0 : i64, "
            "earth.arg_scales = array<f64: 40.125, 40.125>}"))
        direct_plans = []
        for format_ in ("json", "binary"):
            prefix = temp / f"resident-add-{format_}"
            run(direct, prefix.with_suffix(".mlir"), [*common,
                f"--runtime-plan-format={format_}"])
            plan = load(prefix, "test", format_)
            assert plan["format_version"] == 8 and len(plan["external_inputs"]) == 2
            assert len(plan["values"]) == 3 and plan["initialization"] == []
            assert all(v["kind"] == "ciphertext" and v["place"] == PLACE and
                       v["scale_log2"] == 40.125 and "buffer_bytes" not in v and
                       "buffer_layout" not in v for v in plan["values"])
            assert plan["execution"][0]["op"] == "add_cc"
            assert all(op["kind"] != "transfer" for op in plan["execution"])
            direct_plans.append(plan)
        assert direct_plans[0] == direct_plans[1]

        # Host external CT without buffers keeps the existing ApiCall V7.
        host = temp / "host-api.mlir"
        host.write_text(module(
            f'%x = {call("%a, %b", f"{CT}, {CT}", attrs=", runtime.scale_log2 = 40.125 : f64")}\n'
            f'return %x : {CT}', arguments=f"%a: {CT}, %b: {CT}",
            attributes="attributes {earth.arg_scales = array<f64: 40.125, 40.125>}"))
        host_prefix = temp / "host-api"
        run(host, host_prefix.with_suffix(".mlir"), [*common, "--runtime-plan-format=json"])
        host_plan = load(host_prefix, "test", "json")
        assert host_plan["format_version"] == 7
        host_values = {v["id"]: v for v in host_plan["values"]}
        assert all(host_values[i]["place"]["kind"] == "host" for i in host_plan["external_inputs"])

        def reject(name, content, expected, options=()):
            path = temp / f"{name}.mlir"
            path.write_text(content)
            run(path, temp / f"{name}.out.mlir", options, expected)

        for name, value in (("wrong-type", "0 : i32"), ("negative", "-1 : i64"),
                            ("outside-device-count", "1 : i64")):
            reject(f"resident-{name}", module(
                f'%x = {call("%arg0", CT)}\nreturn %x : {CT}', arguments=f"%arg0: {CT}",
                attributes=f"attributes {{runtime.external_inputs_device = {value}}}"),
                ("runtime.external_inputs_device",),
                [f"--assign-ckks-placement=operator-spec={spec_path} device-counts=1"])
        reject("resident-multiple-ranks", module(
            f'%x = {call("%arg0", CT)}\nreturn %x : {CT}', arguments=f"%arg0: {CT}",
            attributes="attributes {runtime.external_inputs_device = 0 : i64}"),
            ("runtime.external_inputs_device", "one rank"),
            [f"--assign-ckks-placement=operator-spec={spec_path} device-counts=1x1"])

        for name, type_, diagnostic in (
            ("negative-bytes", '!ckks.buffer<-1, "layout">', ("buffer", "unsigned 64-bit")),
            ("overflow-bytes", '!ckks.buffer<18446744073709551616, "layout">', ("buffer", "unsigned 64-bit")),
            ("empty-layout", '!ckks.buffer<1234, "">', ("buffer", "layout", "nonempty")),
        ):
            reject(name, module(f'%x = {call(result=type_)}\nreturn %x : {type_}', result=type_), diagnostic)

        for index, type_ in enumerate(("i64", "f64", "tensor<1xi64>", "tensor<1xf64>")):
            reject(f"unsupported-input-{index}", module(
                f'%x = {call("%arg0", type_)}\nreturn %x : {CT}', arguments=f"%arg0: {type_}"),
                ("ckks.apicall", "operand"))
            reject(f"unsupported-result-{index}", module(
                f'%x = {call(result=type_)}\nreturn %x : {type_}', result=type_),
                ("ckks.apicall", "result"))
        for name, type_ in (("plaintext-api-input", PT),
                            ("unranked-api-input", "tensor<*x!ckks.poly<2 * 40 * 5>>")):
            reject(name, module(
                f'%x = {call("%arg0", type_)}\nreturn %x : {CT}', arguments=f"%arg0: {type_}"),
                ("ckks.apicall", "inputs", "ciphertext tensors or buffers"))
        reject("ordinary-buffer-input", module(
            f'%h = {call(result=BUFFER)}\n'
            f'%x = "ckks.addcc"(%h, %h) : ({BUFFER}, {BUFFER}) -> {CT}\nreturn %x : {CT}'),
            ("ckks.addcc", "operand"))
        reject("ordinary-buffer-result", module(
            f'%x = "ckks.addcc"(%arg0, %arg0) : ({CT}, {CT}) -> {BUFFER}\nreturn %x : {BUFFER}',
            result=BUFFER, arguments=f"%arg0: {CT}"), ("ckks.addcc", "result"))

        export = [f"-p=builtin.module(func.func({emit(temp / 'invalid', 'json')}))"]
        reject("external-buffer", module(
            f'%x = {call("%h", BUFFER)}\nreturn %x : {CT}', arguments=f"%h: {BUFFER}"),
            ("buffers", "external inputs"), export)
        reject("final-buffer", module(
            f'%h = {call(result=BUFFER)}\nreturn %h : {BUFFER}', result=BUFFER),
            ("buffers", "final outputs"), export)
        reject("cross-place-buffer", module(
            f'%h = {call(result=BUFFER, attrs=", dist.logical_id = 0 : i64, dist.rank = 0 : i64, dist.device = 0 : i64")}\n'
            f'%x = "ckks.apicall"(%h) {{callee = "consume", call_id = 2 : i64, latency_us = 1 : i64, '
            f'dist.logical_id = 1 : i64, dist.rank = 0 : i64, dist.device = 1 : i64}} : ({BUFFER}) -> {CT}\n'
            f'return %x : {CT}', attributes="attributes {dist.device_counts = array<i64: 2>}"),
            ("cross-place", "buffer", "unsupported"), ["--materialize-ckks-communication"])

        # Preserve the complete uint64 byte count in both formats, without a
        # memory estimate (which correctly cannot add another live allocation).
        maximum = '!ckks.buffer<18446744073709551615, "ckks.hoisted.v1">'
        maximum_path = temp / "maximum.mlir"
        maximum_path.write_text(module(
            f'%h = {call(result=maximum)}\n'
            f'%x = "ckks.apicall"(%h) {{callee = "consume", call_id = 2 : i64, '
            f'latency_us = 1 : i64}} : ({maximum}) -> {CT}\nreturn %x : {CT}'))
        for format_ in ("json", "binary"):
            prefix = temp / f"maximum-{format_}"
            run(maximum_path, prefix.with_suffix(".mlir"), [
                f"-p=builtin.module(func.func(plan-runtime-memory,{emit(prefix, format_)}))"])
            plan = load(prefix, "test", format_)
            buffer = next(v for v in plan["values"] if v["kind"] == "buffer")
            assert buffer["buffer_bytes"] == "18446744073709551615"

        # UINT64_MAX parses correctly; adding a live ciphertext must not wrap.
        reject("memory-overflow", fixture.read_text().replace("1234", "18446744073709551615"),
            ("memory byte sum overflow",),
            [f"--estimate-runtime-memory=prefix={temp / 'overflow'} operator-spec={spec_path}"])
        reject("wrong-argument-scale-type", module(
            f'%x = {call("%arg0", CT)}\nreturn %x : {CT}', arguments=f"%arg0: {CT}",
            attributes="attributes {earth.arg_scales = array<f32: 40.125>}"),
            ("earth.arg_scales", "array<f64>"), export)

        # Ordinary Rescale cannot infer a real scale from a rounded PolyType in
        # the prepared path: require exact output metadata or physical moduli.
        product = "tensor<1x!ckks.poly<2 * 80 * 5>>"
        reduced = "tensor<1x!ckks.poly<2 * 40 * 4>>"
        rescale_body = (
            f'%x = {call(result=product, attrs=", runtime.scale_log2 = 80.125 : f64")}\n'
            f'%r = "ckks.rescalec"(%x) : ({product}) -> {reduced}\n'
            f'return %r : {reduced}')
        reject("unresolved-rescale-scale", module(rescale_body, result=reduced),
               ("ordinary Rescale", "explicit runtime.scale_log2 f64"), export)
        rescale_path = temp / "rescale-exact.mlir"
        rescale_path.write_text(module(rescale_body.replace(
            '"ckks.rescalec"(%x)',
            '"ckks.rescalec"(%x) {runtime.scale_log2 = 40.125 : f64}'), result=reduced))
        rescale_prefix = temp / "rescale-exact"
        run(rescale_path, temp / "rescale-exact.out.mlir", [
            f"-p=builtin.module(func.func({emit(rescale_prefix, 'json')}))"])
        rescale_plan = load(rescale_prefix, "test", "json")
        rescale = next(op for op in rescale_plan["execution"] if op["op"] == "rescale")
        assert rescale["attrs"]["target_scale_log2"] == 40.125
        assert next(v for v in rescale_plan["values"] if v["id"] == rescale["output"])["scale_log2"] == 40.125

        for name, attr in (("integer-scale", "40 : i64"), ("negative-scale", "-1.0 : f64"),
                           ("large-scale", "1024.0 : f64")):
            reject(name, module(f'%x = {call(attrs=f", runtime.scale_log2 = {attr}")}\nreturn %x : {CT}'),
                   ("runtime.scale_log2", "finite f64", "[0, 1023]"))
    print("Prepared buffer V8, metadata, lifetime, memory and verifier checks passed")


if __name__ == "__main__":
    main()
