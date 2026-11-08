"""Build figures for the November progress report from saved experiment outputs."""
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'updates/figures'
summary = json.loads((ROOT / 'results/userbench_executor_e027/summary.json').read_text())
audit = json.loads((ROOT / 'results/e027_audit.json').read_text())
questions = json.loads((ROOT / 'results/userbench_executor_e027/question_outcomes.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none', 'svg.hashsalt': 'progress-2026-11-09', 'axes.spines.top': False, 'axes.spines.right': False})
modes = ['no_information', 'first_only', 'second_only', 'full']
labels = ['Без сведений', 'Первое', 'Второе', 'Оба']
colors = ['#83909c', '#d89540', '#23877c']

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + '.svg'), metadata={'Date': None})
    fig.savefig('/tmp/' + name + '.png', dpi=120)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(4)
for i, (policy, title) in enumerate([('uniform', 'Случайный выбор'), ('lexical_price', 'Эвристика'), ('qwen3_4b', 'Qwen3-4B')]):
    values = [100 * float(Fraction(summary['all_aspects_same_mask_diagnostic'][policy][m]['source_acceptable'])) for m in modes]
    bars = ax.bar(x + (i - 1) * .25, values, .24, label=title, color=colors[i])
    ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=3)
ax.set(xticks=x, xticklabels=labels, ylim=(0, 80), ylabel='Допустимые варианты по исходным меткам, %', xlabel='Какие предпочтения раскрыты для каждой части поездки')
ax.legend(loc='upper left', frameon=False)
save(fig, 'executor_quality')

fig, ax = plt.subplots(figsize=(9, 4.6))
left = np.zeros(4)
for key, title, color in [('T', 'Оба условия выполнены', '#23877c'), ('U', 'Недостаточно сведений', '#d89540'), ('F', 'Есть нарушение', '#b65b57')]:
    values = np.array([audit['by_mask'][m]['semantic_counts'].get(key, 0) for m in modes])
    ax.barh(labels, values, left=left, color=color, label=title)
    for y, (a, v) in enumerate(zip(left, values)):
        if v: ax.text(a + v / 2, y, str(v), ha='center', va='center', color='white')
    left += values
assert np.all(left == 48)
ax.set(xlim=(0, 48), xlabel='Число выборов Qwen, всего 48 в каждом режиме')
ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=3, frameon=False, fontsize=9)
save(fig, 'preference_checks')

fig, ax = plt.subplots(figsize=(9, 4.8))
question_names = ['flight', 'lodging', 'rental_car', 'restaurant']
for i, mode in enumerate(['full', 'first_only', 'second_only']):
    values = [100 * float(Fraction(summary['constant_one_question']['qwen3_4b'][mode][q]['source_acceptable'])) for q in question_names]
    bars = ax.bar(np.arange(4) + (i - 1) * .25, values, .24, label={'full': 'Полный ответ', 'first_only': 'Первое предпочтение', 'second_only': 'Второе предпочтение'}[mode], color=colors[i])
    ax.bar_label(bars, fmt='%.1f', fontsize=9, padding=3)
ax.axhline(100 * 10 / 48, color='#444', linestyle='--', label='Без вопроса, 20,8%')
ax.set(xticks=np.arange(4), xticklabels=['Перелёт', 'Жильё', 'Машина', 'Ресторан'], ylim=(0, 57), ylabel='Средняя доля допустимых частей поездки, %')
ax.legend(frameon=False, fontsize=9, ncol=2, loc='upper left')
save(fig, 'constant_questions')

fig, ax = plt.subplots(figsize=(8, 4.5))
bottom = np.zeros(3)
for number, color in zip([1, 2, 3, 4], ['#23877c', '#7ca9a2', '#d89540', '#b7bdc5']):
    values = []
    for mode in ['full', 'first_only', 'second_only']:
        rows = [r for r in questions if r['policy'] == 'qwen3_4b' and r['mask'] == mode]
        counts = Counter()
        for case in sorted({r['case_id'] for r in rows}):
            gains = [Fraction(r['source_acceptable_gain']) for r in rows if r['case_id'] == case]
            counts[sum(v == max(gains) for v in gains)] += 1
        values.append(counts[number])
    ax.bar(['Полный ответ', 'Первое', 'Второе'], values, bottom=bottom, color=color, label=str(number))
    for i, value in enumerate(values):
        if value: ax.text(i, bottom[i] + value / 2, str(value), ha='center', va='center')
    bottom += values
assert np.all(bottom == 12)
ax.set(ylim=(0, 14), ylabel='Число задач')
ax.legend(title='Вопросов с одинаковым максимумом', ncol=4, fontsize=9, title_fontsize=10, frameon=False)
save(fig, 'question_ties')
print('Four figures saved to', OUT)
