"""Generate Figure 1 (timeline). Events are (decimal year, label lines, lane). Dates: month of release or first paper."""
import sys

def yr(y, m):
    return y + (m - 0.5) / 12.0

LANES = [
    ('Attention and KV cache', 'a', [
        (yr(2019, 11), ['MQA'], 'a'),
        (yr(2020, 5), ['GPT-3: banded', 'and dense layers'], 'a'),
        (yr(2023, 5), ['GQA'], 'a'),
        (yr(2023, 9), ['sliding window', '(Mistral 7B)'], 'a'),
        (yr(2024, 5), ['MLA', '(DeepSeek-V2)'], 'a'),
        (yr(2024, 6), ['local:global 1:1', '(Gemma 2)'], 'a'),
        (yr(2025, 3), ['5:1 (Gemma 3)'], 'a'),
        (yr(2025, 9), ['3:1 linear hybrid', '(Qwen3-Next)'], 'a'),
        (yr(2025, 12), ['sparse top-k', '(V3.2)'], 'a'),
        (yr(2026, 4), ['compressed KV', '(V4)'], 'a'),
        (yr(2026, 9), ['cross-layer', 'KV reuse (V4.1)'], 'a'),
    ]),
    ('FFN and experts', 'b', [
        (yr(2020, 2), ['GLU variants', '(SwiGLU)'], 'b'),
        (yr(2020, 6), ['GShard MoE'], 'b'),
        (yr(2021, 1), ['Switch top-1'], 'b'),
        (yr(2024, 1), ['Mixtral top-2; fine-', 'grained + shared', '(DeepSeekMoE)'], 'b'),
        (yr(2024, 8), ['bias-based', 'balancing'], 'b'),
        (yr(2024, 12), ['V3: 256 experts'], 'b'),
        (yr(2025, 4), ['Qwen3: no', 'shared expert'], 'b'),
        (yr(2025, 7), ['K2: 384', 'experts'], 'b'),
        (yr(2025, 9), ['512 experts', '(Qwen3-Next)'], 'b'),
        (yr(2026, 7), ['LatentMoE, 896', 'experts (K3)'], 'b'),
    ]),
    ('Position, norm, stability', 'c', [
        (yr(2019, 10), ['RMSNorm'], 'c'),
        (yr(2021, 4), ['RoPE'], 'c'),
        (yr(2023, 2), ['LLaMA block;', 'QK-norm (ViT-22B)'], 'c'),
        (yr(2023, 9), ['RoPE base', 'raised, YaRN'], 'c'),
        (yr(2024, 6), ['logit', 'soft-cap'], 'c'),
        (yr(2025, 3), ['QK-norm in', 'Gemma 3, Qwen3'], 'c'),
        (yr(2025, 7), ['QK-Clip (K2)'], 'c'),
        (yr(2025, 8), ['sink logits', '(gpt-oss)'], 'c'),
        (yr(2026, 4), ['mHC residual', '(V4)'], 'c'),
    ]),
]

X0, X1 = 118.0, 706.0
Y_LO, Y_HI = 2019.7, 2026.95
CH = 5.7          # estimated px per character at 11px
LANE_H = 108
TOP = 30

def X(y):
    return X0 + (y - Y_LO) / (Y_HI - Y_LO) * (X1 - X0)

def place(events):
    """greedy row assignment: labels hang below the axis; rows 0..3; returns [(x, lines, row, anchor)]"""
    out = []
    rows_end = {}   # row -> list of (x0, x1)
    for y, lines, lane in sorted(events):
        x = X(y)
        w = max(len(l) for l in lines) * CH + 4
        anchor = 'start'
        x0, x1 = x - 3, x - 3 + w
        if x1 > 716:
            anchor = 'end'
            x0, x1 = x + 3 - w, x + 3
        for r in range(4):
            if all(x0 >= b + 3 or x1 <= a - 3 for a, b in rows_end.get(r, [])):
                rows_end.setdefault(r, []).append((x0, x1))
                out.append((x, lines, r, anchor, y))
                break
        else:
            raise SystemExit(f'no row for {lines} at {y}')
    return out

def build():
    n = len(LANES)
    H = TOP + n * LANE_H + 6
    o = []
    o.append(f'<svg viewBox="0 0 720 {H}" role="img" aria-labelledby="fig-33-1-title">')
    o.append('<title id="fig-33-1-title">Timeline 2019 to 2026 of architectural ideas in open models, in three lanes: attention and KV cache, FFN and experts, position and normalization</title>')
    # year ticks
    for yv in range(2020, 2027):
        x = X(yv)
        o.append(f'<line class="s-line s-dash" x1="{x:.1f}" y1="{TOP - 8}" x2="{x:.1f}" y2="{H - 6}" style="opacity:.45"/>')
        o.append(f'<text x="{x:.1f}" y="{TOP - 12}" text-anchor="middle" class="s-small s-bold">{yv}</text>')
    for i, (title, col, events) in enumerate(LANES):
        y0 = TOP + i * LANE_H
        axis_y = y0 + 14
        o.append(f'<text x="4" y="{axis_y + 4}" class="s-small s-bold s-t{col}">{title.split(" and ")[0] if False else ""}</text>')
        # lane title, two lines
        words = title.replace(' and ', ' and\n').replace(', ', ',\n').split('\n')
        for k, wline in enumerate(words):
            o.append(f'<text x="4" y="{axis_y + 4 + 14 * k}" class="s-small s-bold s-t{col}">{wline}</text>')
        o.append(f'<line class="s-line-{col}" x1="{X0 - 6}" y1="{axis_y}" x2="{X1 + 8}" y2="{axis_y}"/>')
        for x, lines, r, anchor, yv in place(events):
            ly = axis_y + 18 + r * 24
            o.append(f'<circle class="s-fill-{col}" cx="{x:.1f}" cy="{axis_y}" r="3.6"/>')
            o.append(f'<line class="s-line" x1="{x:.1f}" y1="{axis_y + 3}" x2="{x:.1f}" y2="{ly - 10 if len(lines) == 1 else ly - 10}" style="opacity:.55"/>')
            tx = x + 3 if anchor == 'start' else x - 3
            for k, line in enumerate(lines):
                o.append(f'<text x="{tx:.1f}" y="{ly + 14 * k - 0:.1f}" text-anchor="{anchor}" class="s-small">{line}</text>')
    o.append('</svg>')
    return '\n'.join(o), H

if __name__ == '__main__':
    svg, H = build()
    open('/tmp/w33/fig1.svg', 'w').write(svg)
    print('height', H)
