import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import platform
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
if platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

_MA_STYLE = {
    'MA5':  {'color': '#f39c12', 'lw': 1.2},
    'MA20': {'color': '#27ae60', 'lw': 1.4},
    'MA60': {'color': '#8e44ad', 'lw': 1.4},
    'MA120':{'color': '#2980b9', 'lw': 1.2},
}

BULL_COLOR = '#e74c3c'
BEAR_COLOR = '#3498db'


def _draw_candles(ax, df):
    for i, (_, row) in enumerate(df.iterrows()):
        c = BULL_COLOR if row['IsBull'] else BEAR_COLOR
        body_lo = min(row['Open'], row['Close'])
        body_hi = max(row['Open'], row['Close'])
        ax.bar(i, body_hi - body_lo, bottom=body_lo,
               color=c, width=0.6, alpha=0.85, zorder=2)
        ax.plot([i, i], [row['Low'], row['High']],
                color=c, linewidth=0.8, zorder=1)


def _draw_mas(ax, df):
    x = range(len(df))
    for ma, style in _MA_STYLE.items():
        if ma in df.columns:
            ax.plot(x, df[ma], label=ma,
                    color=style['color'], linewidth=style['lw'], alpha=0.9)


def _draw_order_blocks(ax, df, ob_info):
    n = len(df)
    date_list = list(df.index)

    for ob in ob_info.get('bull', [])[-4:]:
        if ob['Date'] not in date_list:
            continue
        start = date_list.index(ob['Date'])
        width = n - start
        rect = mpatches.FancyBboxPatch(
            (start - 0.3, ob['Low']), width + 0.6, ob['High'] - ob['Low'],
            boxstyle='round,pad=0', linewidth=1,
            edgecolor='#27ae60', facecolor='#27ae60', alpha=0.12, zorder=0
        )
        ax.add_patch(rect)
        ax.text(start + 0.5, ob['Mid'], '강세OB',
                fontsize=7, color='#27ae60', alpha=0.9, va='center')

    for ob in ob_info.get('bear', [])[-4:]:
        if ob['Date'] not in date_list:
            continue
        start = date_list.index(ob['Date'])
        width = n - start
        rect = mpatches.FancyBboxPatch(
            (start - 0.3, ob['Low']), width + 0.6, ob['High'] - ob['Low'],
            boxstyle='round,pad=0', linewidth=1,
            edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.12, zorder=0
        )
        ax.add_patch(rect)
        ax.text(start + 0.5, ob['Mid'], '약세OB',
                fontsize=7, color='#e74c3c', alpha=0.9, va='center')


def _draw_signals(ax, df):
    for i, (_, row) in enumerate(df.iterrows()):
        lo, hi = row['Low'], row['High']
        if row.get('SE_Entry', False):
            ax.scatter(i, lo * 0.990, marker='^', color='#27ae60', s=120, zorder=5)
        if row.get('SE_Rally', False):
            ax.scatter(i, lo * 0.984, marker='^', color='#f39c12', s=160, zorder=5)
        if row.get('SE_BearTrap', False):
            ax.scatter(i, lo * 0.993, marker='D', color='#9b59b6', s=90, zorder=5)
        if row.get('SE_Exit', False):
            ax.scatter(i, hi * 1.010, marker='v', color='#e74c3c', s=120, zorder=5)
        if row.get('SE_Accum', False):
            ax.scatter(i, lo * 0.997, marker='s', color='#1abc9c', s=60, zorder=5)
        if row.get('Pullback', False):
            ax.scatter(i, lo * 0.995, marker='o', color='#3498db', s=70, zorder=5)


def _signal_legend():
    return [
        mpatches.Patch(color='#27ae60', label='▲ 세력 진입'),
        mpatches.Patch(color='#f39c12', label='▲ 상승 시작'),
        mpatches.Patch(color='#9b59b6', label='◆ 베어트랩 회복'),
        mpatches.Patch(color='#e74c3c', label='▼ 세력 이탈'),
        mpatches.Patch(color='#1abc9c', label='■ 세력 매집'),
        mpatches.Patch(color='#3498db', label='● 눌림목'),
    ]


def plot(df: pd.DataFrame, ob_info: dict,
         ticker: str, name: str,
         pts: int, tags: list,
         bars: int = 120,
         save: bool = True):

    view = df.iloc[-bars:].copy() if len(df) > bars else df.copy()
    n = len(view)

    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1.5, 1], hspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ── 캔들 + 이평선 + OB + 신호 ────────────────────────────────────────
    _draw_candles(ax1, view)
    _draw_mas(ax1, view)
    _draw_order_blocks(ax1, view, ob_info)
    _draw_signals(ax1, view)

    tag_str = '  |  '.join(tags[:4])
    ax1.set_title(f'{name} ({ticker})   점수: {pts}점\n{tag_str}',
                  fontsize=12, fontweight='bold', pad=8)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)

    sig_legend = fig.legend(handles=_signal_legend(),
                             loc='upper right', fontsize=8,
                             bbox_to_anchor=(0.99, 0.97))
    ax1.add_artist(sig_legend)
    ax1.grid(alpha=0.25)
    ax1.set_xlim(-1, n)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── 거래량 바 ─────────────────────────────────────────────────────────
    for i, (_, row) in enumerate(view.iterrows()):
        c = BULL_COLOR if row['IsBull'] else BEAR_COLOR
        ax2.bar(i, row['Volume'], color=c, width=0.6, alpha=0.7)
    if 'VolMA' in view.columns:
        ax2.plot(range(n), view['VolMA'], color='orange', linewidth=1.4, label='거래량 MA20')
        ax2.plot(range(n), view['VolMA'] * 3, color='red',
                 linewidth=0.9, linestyle='--', alpha=0.6, label='3x 기준')
    ax2.set_ylabel('거래량', fontsize=9)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(alpha=0.25)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── 거래량 배율 ───────────────────────────────────────────────────────
    vol_r = view['VolRatio'].fillna(0)
    colors = ['#e74c3c' if v >= 3 else '#95a5a6' for v in vol_r]
    ax3.bar(range(n), vol_r, color=colors, width=0.6, alpha=0.85)
    ax3.axhline(3, color='red', linewidth=0.9, linestyle='--', alpha=0.7)
    ax3.axhline(1, color='gray', linewidth=0.5, alpha=0.5)
    ax3.set_ylabel('거래량\n배율(x)', fontsize=8)

    step = max(1, n // 10)
    ticks = list(range(0, n, step))
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(
        [str(view.index[t].date()) for t in ticks],
        rotation=40, fontsize=7
    )
    ax3.grid(alpha=0.25)
    ax3.set_xlim(-1, n)

    plt.tight_layout()
    if save:
        fname = f'{ticker}_{name}_분석.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'차트 저장: {fname}')
    plt.show()
