#!/usr/bin/env python3
"""Render complete original/reference/compiler DAGs using local Graphviz.

Outputs self-contained offline HTML, individual SVG/DOT, and combined SVG/DOT.
No graph operations are hidden. Correspondence labels use structural identity,
not SSA ids or a claimed proof of general algebraic equivalence.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from verify import read_json
from reference import plan_bytes, reference_plan


SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
TITLES = {"original": "Original", "manual": "Reference", "greedy": "Greedy",
          "depth-dp": "Depth-DP", "optimized": "Optimized"}
COLORS = {
    "input": ("#e0f2fe", "#0369a1"), "output": ("#dcfce7", "#15803d"),
    "constant": ("#f1f5f9", "#64748b"), "compute": ("#eff6ff", "#3b82f6"),
    "boot": ("#fee2e2", "#dc2626"), "rescale": ("#ffedd5", "#ea580c"),
    "mod_switch": ("#f3e8ff", "#9333ea"), "align": ("#fae8ff", "#c026d3"),
    "relinearize": ("#e0e7ff", "#6366f1"), "failed": ("#fff1f2", "#be123c"),
}


def fingerprint(op, operands=(), attrs=None):
    if op in ("mul", "add"):
        operands = sorted(operands)
    return hashlib.sha256(json.dumps([op, list(operands), attrs], sort_keys=True).encode()).hexdigest()


def constant_fingerprint(values):
    return fingerprint("constant", attrs=[float(x).hex() for x in values])


def fmt(value):
    return f"{value:.4f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)


def state_label(value):
    return f"s={fmt(value['scale_log2'])}  L={value['level']}  c={value['components']}"


def constant_label(values):
    if all(x == values[0] for x in values):
        return f"{fmt(values[0])} x {len(values)} slots"
    text = ", ".join(fmt(v) for v in values[:3])
    return f"[{text}{', ...' if len(values) > 3 else ''}]"


def node(key, label, kind, origins, detail, dead=False):
    return dict(key=key, label=label, kind=kind, origins=origins, detail=detail, dead=dead)


def operand_edges(operands, target):
    return [dict(source=value, target=target, count=count)
            for value, count in Counter(operands).items()]


def original_model(graph, case, spec):
    slots = spec["context"]["poly_degree"] // 2
    fingerprints, origins = {}, defaultdict(list)
    nodes, edges = [], []
    needed = set(graph["outputs"])
    for row in reversed(graph["ops"]):
        name, _, operands = row[:3]
        if name in needed:
            needed.update(operands)
    for index, (name, state) in enumerate(zip(case["input_names"], case["inputs"])):
        fingerprints[name] = fingerprint("input", attrs=index)
        origins[fingerprints[name]].append(name)
        nodes.append(node(name, f"INPUT {name}\n{state_label(state)}", "input", [name],
                          dict(original=name, state=state), name not in needed))
    for name, coefficient in graph["constants"].items():
        values = coefficient if isinstance(coefficient, list) else [coefficient] * slots
        fingerprints[name] = constant_fingerprint(values)
        origins[fingerprints[name]].append(name)
        nodes.append(node(name, f"CONST {name}\n{constant_label(values)}", "constant",
                          [name], dict(original=name, coefficient=coefficient), name not in needed))
    for row in graph["ops"]:
        name, op, operands = row[:3]
        shift = row[3]["steps"] if len(row) == 4 and op == "rotate" else 1
        fp = fingerprint(op, [fingerprints[v] for v in operands], shift % slots if op == "rotate" else None)
        fingerprints[name] = fp
        origins[fp].append(name)
        nodes.append(node(name, f"{name}  {op.upper()}" + (f" {shift:+d}" if op == "rotate" else ""),
                          "compute", [name], dict(original=name, op=op, inputs=operands), name not in needed))
        edges.extend(operand_edges(operands, name))
    for index, value in enumerate(graph["outputs"]):
        name = f"return:{index}"
        nodes.append(node(name, f"RETURN {index}\n{value}", "output", [value],
                          dict(original=value, contract=case["outputs"][index])))
        edges.append(dict(source=value, target=name, count=1))
    return dict(nodes=nodes, edges=edges, boot_count=None, counts=Counter(row[1] for row in graph["ops"])), origins


def plan_model(plan, case, spec, origins):
    desc = {v["id"]: v for v in plan["values"]}
    slots = spec["context"]["poly_degree"] // 2
    nodes, edges, fingerprints, constant_values = [], [], {}, {}
    counts = Counter()
    for index, value in enumerate(plan["external_inputs"]):
        fp = fingerprint("input", attrs=index)
        fingerprints[value] = fp
        name = case["input_names"][index]
        nodes.append(node(value, f"INPUT {name}  %{value}\n{state_label(desc[value])}", "input",
                          origins.get(fp, []), dict(value=desc[value], external_input=index)))
    for phase in ("initialization", "execution", "finalization"):
        for step in plan[phase]:
            out = step["output"]
            if step["kind"] == "encode":
                if step["payload"]["kind"] != "inline":
                    raise ValueError("graph renderer requires inline plaintext")
                values = list(step["payload"]["values"])
                values += [0.] * (slots - len(values))
                constant_values[out] = values
                fp = constant_fingerprint(values)
                kind = "constant"
                title = f"ENCODE  %{out}\n{constant_label(values)}"
                counts["encode"] += 1
            elif step["kind"] == "compute":
                op, operands = step["op"], step["inputs"]
                counts[op] += 1
                if not all(value in fingerprints for value in operands):
                    raise ValueError("cannot render use before definition")
                fps = [fingerprints[v] for v in operands]
                kind = op if op in COLORS else "compute"
                if op in ("boot", "rescale", "mod_switch", "relinearize"):
                    fp = fps[0]
                elif op == "mul_cp" and constant_values.get(operands[1]) == [1.] * slots:
                    fp, kind = fps[0], "align"
                elif op.startswith(("mul_", "add_", "sub_")):
                    fp = fingerprint(op.split("_")[0], fps)
                elif op == "rotate":
                    fp = fingerprint("rotate", fps, step["attrs"]["steps"] % slots)
                elif op == "negate":
                    fp = fingerprint("negate", fps)
                else:
                    raise ValueError(f"unsupported graph operation: {op}")
                title = f"{op.upper()}  %{out}"
                if kind == "align":
                    title += "\nscale alignment (x 1)"
                if op == "rotate":
                    title += f"  {step['attrs']['steps']:+d}"
                edges.extend(operand_edges(operands, out))
            else:
                raise ValueError(f"unsupported graph step: {step['kind']}")
            fingerprints[out] = fp
            matched = origins.get(fp, [])
            if matched:
                title += "\n[" + ", ".join(matched) + "]"
            title += "\n" + state_label(desc[out])
            nodes.append(node(out, title, kind, matched,
                dict(phase=phase, step=step, output_state=desc[out],
                     input_states=[desc[v] for v in step.get("inputs", [])],
                     structural_correspondence=matched)))
    for index, value in enumerate(plan["final_outputs"]):
        key = f"return:{index}"
        nodes.append(node(key, f"RETURN {index}\n%{value}\n{state_label(desc[value])}", "output",
                          origins.get(fingerprints[value], []), dict(value=desc[value], contract=case["outputs"][index])))
        edges.append(dict(source=value, target=key, count=1))
    return dict(nodes=nodes, edges=edges, boot_count=counts["boot"], counts=counts)


def dot_body(model, prefix):
    q = lambda text: json.dumps(text, ensure_ascii=False)
    ids = {n["key"]: f"{prefix}-n{i}" for i, n in enumerate(model["nodes"])}
    rows = []
    for n in model["nodes"]:
        fill, border = COLORS[n["kind"]]
        style = "rounded,filled,dashed" if n["dead"] else "rounded,filled"
        shape = "ellipse" if n["kind"] in ("input", "output") else "box"
        rows.append(f'{q(ids[n["key"]])} [id={q(ids[n["key"]])}, label={q(n["label"])}, '
                    f'shape={shape}, style={q(style)}, fillcolor={q(fill)}, color={q(border)}, '
                    f'fontcolor={q("#94a3b8" if n["dead"] else "#172033")}, '
                    f'penwidth={2.5 if n["kind"] == "boot" else 1.3}];')
    kinds = {n["key"]: n["kind"] for n in model["nodes"]}
    consumers = Counter(e["source"] for e in model["edges"])
    for e in model["edges"]:
        # Prefer straight ciphertext chains; constants must not push a long
        # chain diagonally across the entire canvas. All edges remain visible.
        weight = 10 if kinds[e["source"]] != "constant" and consumers[e["source"]] == 1 else 1
        label = f', label="x{e["count"]}"' if e["count"] > 1 else ""
        rows.append(f'{q(ids[e["source"]])} -> {q(ids[e["target"]])} [weight={weight}{label}];')
    return "\n".join(rows), ids


DOT_STYLE = ('graph [rankdir=TB, bgcolor="white", nodesep=0.25, ranksep=0.42, pad=0.2];\n'
             'node [fontname="Arial", fontsize=12, margin="0.13,0.08"];\n'
             'edge [fontname="Arial", fontsize=10, color="#94a3b8", arrowsize=0.65];\n')


def render_svg(dot, model, prefix, directory, stem, executable):
    dot_path = directory / f"{stem}.dot"
    dot_path.write_text(dot)
    run = subprocess.run([executable, "-Tsvg", str(dot_path)], capture_output=True, text=True, timeout=60)
    if run.returncode:
        raise RuntimeError(run.stderr)
    root = ET.fromstring(run.stdout)
    root.set("role", "img")
    root.set("aria-label", stem)
    _, ids = dot_body(model, prefix)
    by_id = {ids[n["key"]]: n for n in model["nodes"]}
    for group in root.iter(f"{{{SVG}}}g"):
        if group.get("id") not in by_id:
            continue
        n = by_id[group.get("id")]
        group.set("data-origins", json.dumps(n["origins"]))
        group.set("data-kind", n["kind"])
        group.set("data-detail", json.dumps(n["detail"], ensure_ascii=False))
        group.set("tabindex", "0")
        title = group.find(f"{{{SVG}}}title")
        if title is not None:
            title.text = json.dumps(n["detail"], indent=2, ensure_ascii=False)
    width, height = [float(x) for x in root.get("viewBox").split()[2:]]
    root.set("data-width", str(width))
    root.set("data-height", str(height))
    text = ET.tostring(root, encoding="unicode")
    (directory / f"{stem}.svg").write_text(text)
    return text


CSS = """
:root{font-family:system-ui,-apple-system,sans-serif;color:#172033;background:#f6f8fc}*{box-sizing:border-box}
body{margin:0}header{padding:22px 28px 14px;background:white;border-bottom:1px solid #dbe2ec}
h1{font-size:23px;margin:10px 0}p{line-height:1.65}a{color:#245ec2}button,select{font:inherit;padding:5px 9px;cursor:pointer}
.muted{color:#64748b;font-size:13px}.toolbar,.legend{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:12px 0}
.legend span::before{content:'';display:inline-block;width:11px;height:11px;margin-right:5px;border-radius:3px;background:var(--color)}
.boards{display:flex;gap:12px;overflow-x:auto;padding:16px 20px;align-items:flex-start}
.panel{flex:1 0 280px;min-width:280px;background:white;border:1px solid #dbe2ec;border-radius:10px;overflow:hidden}
.panel[hidden]{display:none}.panel h2{font-size:16px;margin:0;padding:12px 14px 5px}.stats{font-size:12px;color:#475569;padding:0 14px 10px;min-height:30px}
.badge{display:inline-block;padding:2px 7px;border-radius:4px;background:#dcfce7;color:#166534;font-size:12px;margin-left:7px}
.badge.bad{background:#ffe4e6;color:#9f1239}.viewport{height:66vh;overflow:auto;border-top:1px solid #edf1f7;padding:8px}
.viewport svg{display:block;margin:0 auto;max-width:none}.node{cursor:pointer;transition:opacity .15s}
.node.selected polygon,.node.selected path,.node.selected ellipse{stroke:#0f172a!important;stroke-width:3.5px!important}
body.focus-management .node[data-kind=compute],body.focus-management .node[data-kind=constant]{opacity:.24}
.node.dim{opacity:.16!important}.node.selected{opacity:1!important}.failure{padding:16px;color:#9f1239;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
details{margin:0 20px 20px;background:white;border:1px solid #dbe2ec;border-radius:8px;padding:12px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:300px;overflow:auto}
table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px 14px;border-bottom:1px solid #e2e8f0;font-size:14px}th{color:#475569}
.index{max-width:1200px;margin:auto;padding:20px}.index input{font:inherit;padding:8px;width:320px;max-width:100%}
@media(max-width:700px){header{padding:14px}.boards{padding:12px}.panel{flex-basis:85vw}.viewport{height:65vh}}
"""


JS = """
const allNodes=[...document.querySelectorAll('.node[data-origins]')];
const zoom=document.querySelector('#zoom');
function setZoom(){const z=Number(zoom.value)/100;document.querySelector('#zoom-label').textContent=zoom.value+'%';
 document.querySelectorAll('.viewport svg').forEach(s=>{s.style.width=(Number(s.dataset.width)*z)+'px';s.style.height=(Number(s.dataset.height)*z)+'px';});}
zoom.addEventListener('input',setZoom);setZoom();
// Deep graphs can start far to the right of their canvas. Start each column
// at its first ciphertext input instead of showing an empty left margin.
requestAnimationFrame(()=>document.querySelectorAll('.panel:not([hidden]) .viewport').forEach(v=>{
 const n=v.querySelector('.node[data-kind=input]');if(!n)return;
 const r=n.getBoundingClientRect(),p=v.getBoundingClientRect();
 v.scrollLeft+=r.left+r.width/2-p.left-v.clientWidth/2;
}));
document.querySelector('#fit').onclick=()=>{const ratios=[...document.querySelectorAll('.panel:not([hidden]) .viewport svg')].map(s=>(s.parentElement.clientWidth-22)/Number(s.dataset.width));
 if(ratios.length){zoom.value=Math.max(20,Math.min(200,Math.floor(Math.min(...ratios)*100)));setZoom();}};
document.querySelector('#management').onchange=e=>document.body.classList.toggle('focus-management',e.target.checked);
function clear(){allNodes.forEach(n=>n.classList.remove('dim','selected'));document.querySelector('#selection').textContent='点击节点可跨方案高亮对应计算';}
document.querySelector('#clear').onclick=clear;
allNodes.forEach(n=>{function select(){const names=JSON.parse(n.dataset.origins);allNodes.forEach(x=>{const hit=x===n||JSON.parse(x.dataset.origins).some(k=>names.includes(k));x.classList.toggle('selected',hit);x.classList.toggle('dim',!hit);});
 document.querySelector('#selection').textContent=names.length?'对应原始值：'+names.join(', '):'无可确认的结构对应；仅高亮当前节点';
 document.querySelector('#detail').textContent=JSON.stringify(JSON.parse(n.dataset.detail),null,2);document.querySelector('#details').open=true;}
 n.onclick=select;n.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};});
let syncing=false;document.querySelectorAll('.viewport').forEach(v=>v.addEventListener('scroll',()=>{
 if(syncing||!document.querySelector('#sync').checked)return;syncing=true;
 const fraction=v.scrollTop/Math.max(1,v.scrollHeight-v.clientHeight);
 document.querySelectorAll('.viewport').forEach(other=>{if(other!==v)other.scrollTop=fraction*Math.max(0,other.scrollHeight-other.clientHeight);});
 requestAnimationFrame(()=>syncing=false);
}));
document.querySelectorAll('[data-panel-toggle]').forEach(c=>c.onchange=()=>{document.getElementById('panel-'+c.dataset.panelToggle).hidden=!c.checked;});
"""


def document(title, body, script=""):
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style><body>{body}'
            f'<script>{script}</script></body></html>')


def graph_caption(model):
    c = model["counts"]
    if model["boot_count"] is None:
        return f"{sum(c.values())} original compute ops"
    return f"Boot {c['boot']} | Rescale {c['rescale']} | ModSwitch {c['mod_switch']} | Relin {c['relinearize']}"


def contract_caption(spec):
    p = spec["boot_profiles"][0]
    input_scale = f"s={p['input_scale_log2']}" if "input_scale_log2" in p else "s=UNRESTRICTED"
    return (f"Boot: L=[{p['input_level_min']},{p['input_level_max']}], {input_scale} -> "
            f"L={p['output_level']}, s={p['output_scale_log2']}; "
            f"arithmetic L >= {spec['levels']['lower_bound']}; s=log2(scale)")


def combined_svg(panels, destination, contract):
    """Tile already rendered SVGs so Graphviz packing cannot drop edge labels."""
    tiles = []
    for key, status, model, _ in panels:
        if model is None:
            tiles.append((key, status, None, 360., 150., "No generated plan"))
            continue
        svg = ET.parse(destination / f"{key}.svg").getroot()
        width, height = [float(x) for x in svg.get("viewBox").split()[2:]]
        caption = graph_caption(model)
        column = max(width + 24, len(caption) * 6.5 + 24, 320)
        tiles.append((key, status, svg, column, height + 86, caption))
    width = sum(t[3] for t in tiles) + 16 * (len(tiles) + 1)
    height = max(t[4] for t in tiles) + 94
    root = ET.Element(f"{{{SVG}}}svg", dict(viewBox=f"0 0 {width} {height}",
                      width=f"{width}pt", height=f"{height}pt", role="img"))
    def element(parent, tag, **attrs):
        return ET.SubElement(parent, f"{{{SVG}}}{tag}", {k: str(v) for k, v in attrs.items()})
    def text(x, y, value, size=14, color="#172033"):
        el = element(root, "text", x=x, y=y, fill=color)
        el.set("font-family", "Arial, sans-serif")
        el.set("font-size", str(size))
        el.set("text-anchor", "middle")
        el.text = value
    element(root, "rect", width=width, height=height, fill="white")
    text(width / 2, 25, contract, size=14)
    x = 16
    for key, status, svg, column, tile_height, caption in tiles:
        element(root, "rect", x=x, y=52, width=column, height=tile_height, rx=10,
                fill="white", stroke="#cbd5e1")
        text(x + column / 2, 76, f"{TITLES[key]} [{status}]", color="#be123c" if status not in ("source", "valid") else "#172033")
        text(x + column / 2, 97, caption, size=12)
        if svg is not None:
            w, h = [float(v) for v in svg.get("viewBox").split()[2:]]
            svg.set("x", str(x + (column - w) / 2))
            svg.set("y", "112")
            svg.set("width", str(w))
            svg.set("height", str(h))
            root.append(svg)
        else:
            text(x + column / 2, 150, "Compile failed / timed out", size=12, color="#be123c")
        x += column + 16
    text(width / 2, height - 10, "s = log2(scale); L = modulus-chain index (L+1 moduli); c = components; x2 = repeated operand", size=12)
    (destination / "comparison.svg").write_text(ET.tostring(root, encoding="unicode"))


def render_case(directory, destination, row, spec, results_dir, executable):
    graph, case = read_json(directory / "graph.json"), read_json(directory / "case.json")
    original, origins = original_model(graph, case, spec)
    manual = plan_model(reference_plan(directory), case, spec, origins)
    panels = [("original", "source", original, ""), ("manual", "valid", manual, "")]
    for strategy, result in row["algorithms"].items():
        status = result["status"]
        model = None
        if status in ("valid", "invalid"):
            path = results_dir / row["case"] / f"{strategy}.graph.runtime-plan.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != result["plan_sha256"]:
                raise ValueError(f"{row['case']}/{strategy}: automatic plan changed after verification")
            model = plan_model(read_json(path), case, spec, origins)
        elif status not in ("compile_failed", "timeout"):
            raise ValueError(f"unexpected result status: {status}")
        panels.append((strategy, status, model, result.get("error", "Compiler timed out" if status == "timeout" else "")))
    destination.mkdir(parents=True, exist_ok=True)
    cards, toggles, clusters = [], [], []
    model_metadata = {}
    for key, status, model, error in panels:
        title = TITLES[key]
        label = {"source": "原始图", "valid": "校验通过", "invalid": "未通过校验",
                 "compile_failed": "编译失败", "timeout": "超时"}[status]
        badge = f'<span class="badge{" bad" if status not in ("source", "valid") else ""}">{label}</span>'
        visible = key != "original" or len(panels) <= 2
        toggles.append(f'<label><input type="checkbox" {"checked" if visible else ""} data-panel-toggle="{key}"> {title}</label>')
        if model is not None:
            caption = graph_caption(model)
            body, _ = dot_body(model, key)
            dot = "digraph G {\n" + DOT_STYLE + body + "\n}\n"
            svg = render_svg(dot, model, key, destination, key, executable)
            content = svg
            links = f' · <a href="{key}.svg">SVG</a> · <a href="{key}.dot">DOT</a>'
            model_metadata[key] = dict(status=status, node_count=len(model["nodes"]), edge_count=len(model["edges"]),
                operation_counts=dict(model["counts"]), unmatched_compute_nodes=[n["key"] for n in model["nodes"]
                    if n["kind"] == "compute" and not n["origins"]])
        else:
            caption = "No generated plan; no comparison count"
            content, links = "", ""
            placeholder = dict(nodes=[node("failed", title + "\n" + status, "failed", [], {})], edges=[])
            body, _ = dot_body(placeholder, key)
            model_metadata[key] = dict(status=status, node_count=0, edge_count=0)
        cluster_label = f"{title}  [{status}]\n{caption}"
        clusters.append(f'subgraph "cluster_{key}" {{label={json.dumps(cluster_label)}; '
                        f'fontname="Arial"; fontsize=16; labelloc=t; sortv={len(clusters)}; style="rounded"; color="#cbd5e1";\n{body}\n}}')
        if error:
            content = f'<div class="failure">{html.escape(error)}</div>' + content
        cards.append(f'<section class="panel" id="panel-{key}" {"" if visible else "hidden"}><h2>{title}{badge}</h2>'
                     f'<div class="stats">{html.escape(caption)}{links}</div><div class="viewport">{content}</div></section>')
    combined = ('digraph comparison {\n' + DOT_STYLE +
                f'graph [pack=true, packmode="array_ut{len(panels)}", label={json.dumps(contract_caption(spec))}, '
                'labelloc=b, fontname="Arial"];\n' + "\n".join(clusters) + "\n}\n")
    (destination / "comparison.dot").write_text(combined)
    combined_svg(panels, destination, contract_caption(spec))
    legend = ''.join(f'<span style="--color:{COLORS[k][1]}">{text}</span>' for k, text in
                     (("boot", "Boot"), ("rescale", "Rescale"), ("mod_switch", "ModSwitch"),
                      ("align", "乘以 1 对齐 scale"), ("relinearize", "Relinearize"), ("compute", "原始计算")))
    initial = "; ".join(f"{name}: L={v['level']}, s={v['scale_log2']}" for name, v in zip(case["input_names"], case["inputs"]))
    bounds = "; ".join(f"L≥{v['min_level']}, s∈[{v['min_scale_log2']},{v['max_scale_log2']}]" for v in case["outputs"])
    body = (f'<header><a href="../index.html">← 全部 case</a><h1>{html.escape(row["case"])} · 计算图对比</h1>'
            f'<p><strong>{html.escape(contract_caption(spec))}</strong></p>'
            f'<p>初始：{html.escape(initial)}。输出：{html.escape(bounds)}。'
            f'单位明文最低 s={case["min_unit_plaintext_scale_log2"]}。</p>'
            '<div class="muted">s = log₂(scale)，L 是模数链索引（保留 L+1 个模数），c = 密文组件数。节点展示值经过四位小数舍入；点击查看精确值。'
            '勾选 Original 查看原始图；虚线表示不影响输出的分支。x2 边表示同一操作数使用两次。</div>'
            f'<div class="legend">{legend}</div><div class="toolbar">'
            '<label>缩放 <input id="zoom" type="range" min="20" max="200" value="100" step="5"> <span id="zoom-label"></span></label>'
            '<button id="fit">适应列宽</button><label><input id="sync" type="checkbox"> 同步纵向滚动</label>'
            '<label><input id="management" type="checkbox"> 突出管理操作</label><button id="clear">清除高亮</button>'
            '<a href="comparison.svg">整组 SVG</a></div><div class="toolbar">' + ''.join(toggles) +
            '</div><div id="selection" class="muted">点击节点可跨方案高亮对应计算</div></header>'
            '<main class="boards">' + ''.join(cards) + '</main>'
            '<details id="details"><summary>选中节点的精确状态与指令</summary><pre id="detail">尚未选择节点</pre></details>'
            '<p class="muted" style="padding:0 24px">[v0] 等标签依据原始表达式结构匹配；管理操作保持对应值，'
            '全 slot 编码的 1 用于识别 scale 对齐。未匹配不代表计算错误，也不会猜测对应关系。'
            '校验未通过的图仅供诊断，其 Boot 数不参与合法方案优劣比较。</p>')
    (destination / "index.html").write_text(document(row["case"], body, JS))
    (destination / "graph-metadata.json").write_text(json.dumps(model_metadata, indent=2) + "\n")
    return model_metadata


def render_suite(suite, results_dir, output_dir):
    suite, results_dir, output_dir = Path(suite).resolve(), Path(results_dir).resolve(), Path(output_dir).resolve()
    executable = shutil.which("dot")
    if executable is None:
        raise RuntimeError("Graphviz 'dot' is required for --graphs; install it explicitly before rendering")
    summary = read_json(results_dir / "summary.json")
    spec_path = suite / "operator-spec.json"
    if summary["operator_spec_sha256"] != "sha256:" + hashlib.sha256(spec_path.read_bytes()).hexdigest():
        raise ValueError("OperatorSpec changed after the comparison run")
    spec = read_json(spec_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    table, rendered = [], {}
    strategies = list(dict.fromkeys(s for row in summary["cases"] for s in row["algorithms"]))
    for row in summary["cases"]:
        directory = suite / row["case"]
        for name in ("graph", "case"):
            path = directory / f"{name}.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["artifacts_sha256"][name]:
                raise ValueError(f"{row['case']}: {name} changed after verification")
        if hashlib.sha256(plan_bytes(reference_plan(directory))).hexdigest() != row["artifacts_sha256"]["manual"]:
            raise ValueError(f"{row['case']}: manual changed after verification")
        rendered[row["case"]] = render_case(directory, output_dir / row["case"], row, spec, results_dir, executable)
        cells = [f'<td><a href="{html.escape(row["case"], quote=True)}/index.html">{html.escape(row["case"])}</a></td>',
                 f'<td>{row["input_compute_ops"]}</td>', f'<td>{row["reference_boots"]}</td>']
        for strategy in strategies:
            result = row["algorithms"].get(strategy)
            if result is None:
                value = "未运行"
            elif result["status"] == "valid":
                value = str(result["stats"]["boot_count"])
            else:
                value = {"invalid": "未通过校验", "compile_failed": "编译失败", "timeout": "超时"}[result["status"]]
            cells.append(f'<td>{value}</td>')
        table.append('<tr>' + ''.join(cells) + '</tr>')
    headings = '<th>case</th><th>原始计算数</th><th>参考 Boot</th>' + ''.join(f'<th>{TITLES[s]} Boot</th>' for s in strategies)
    compiler = summary.get("compiler")
    provenance = f'编译器 SHA-256：{compiler["sha256"]}' if compiler else "仅校验参考方案，未运行编译器"
    body = (f'<main class="index"><h1>{html.escape(summary["suite"])} · 计算图对比</h1>'
            f'<p>{len(table)} 个 case。点击名称查看原始图、完整参考方案及已运行的编译方案。</p>'
            f'<p><strong>{html.escape(contract_caption(spec))}</strong></p>'
            '<p class="muted">参考方案未普遍证明全局最优。所有图保留完整操作和依赖；非法方案仅用于诊断。'
            '浏览页不联网，可复制整个目录离线查看。</p>'
            f'<p class="muted">{html.escape(provenance)}</p><p><input id="filter" placeholder="筛选 case 名称" aria-label="筛选 case"></p>'
            f'<table><thead><tr>{headings}</tr></thead><tbody>{"".join(table)}</tbody></table></main>')
    script = "document.querySelector('#filter').oninput=e=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()));"
    (output_dir / "index.html").write_text(document(summary["suite"], body, script))
    (output_dir / "graph-metadata.json").write_text(json.dumps(rendered, indent=2) + "\n")
    return output_dir / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(render_suite(args.suite, args.results_dir, args.output_dir))


if __name__ == "__main__":
    main()
