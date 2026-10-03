# Perception transition：既存10 seedの追加解析

入力は `data/cohort_replicates/evolving_seed_0_cohorts.csv`〜`evolving_seed_9_cohorts.csv` とそのmetadataのみ。10 seedとも tick 0〜5000、各tickにp0〜p8の9行が揃っている。非空cohortの対象平均値に欠損はない。設定はseedを除いて一致する。新しいsimulationは実行しておらず、入力CSV・Config・Brain・RNG・生態ルールを変更していない。

## 集計の定義

- 対象はp0/p1/p2、区間は (0,1000], (1000,2000], (2000,3000], (3000,4000], (4000,5000]。tick 0は区間に含めない。
- **population share**：各tickの `living_population_p / 全pのliving_population合計` を、seed・区間内で時間平均。cohort不在は0、全体絶滅はNaN。区間内の人口総和の比とは異なる。
- **繁殖参加率**：区間内 `successful_parent_participations` 合計 / `action_opportunities` 合計。出生1件には両親の参加2件があるため、出生率ではない。
- **死亡率**：区間内 `deaths` 合計 / `action_opportunities` 合計。
- **hunger/thirst/signal、age、generation**：区間内の `Σ(cohort平均 × living_population) / Σ(living_population)`。生存個体tickで重み付けした平均である。空cohortは平均に含めない。CSVには形質ごとの有効tick数・有効生存exposureも保存する。
- 生存snapshotは新生児を含み、そのtickの死亡個体を除く。action opportunityは新生児を除き、行動後に死亡した個体を含む。したがって、形質平均の対象と率の分母は厳密には同一ではない。
- これらを**各seed内で先に計算**し、その10個の値を等重みで mean、sample SD (`ddof=1`)、min、max、n に集計。seedを横断して件数・exposureをpoolしていない。
- p0−p2差は同じseed・同じ区間の値を引き算してから集計する。正・負・厳密な0のseed数を別々に数える。有意差検定ではない。分母0・欠測はNaN（CSVでは空欄）とし、nから除外する。
- exposureの大きさによる除外閾値は置いていない。各図に有効seed数nとaction exposureの中央値[min,max]を併記し、CSVに個別値を残した。
- cohort別のmemory size、learning updates、Q値、学習経験量は入力に記録されていない。全体平均などから代用・推定していない。

## 10 seedで再現したこと

(1000,2000]、(2000,3000]、(3000,4000]の3区間では、**p0の繁殖参加率>p2、死亡率<p2が同時に10/10 seedで再現**した。率はすべてaction opportunity当たり。

| 区間 | p0−p2 繁殖参加率 mean ± SD | 正/負/0 | p0−p2 死亡率 mean ± SD | 正/負/0 | 有効n |
|---|---:|---:|---:|---:|---:|
| (0,1000] | −0.001385 ± 0.003794 | 6/4/0 | +0.000747 ± 0.002737 | 6/4/0 | 10 |
| (1000,2000] | +0.002021 ± 0.000311 | 10/0/0 | −0.002146 ± 0.000345 | 0/10/0 | 10 |
| (2000,3000] | +0.001780 ± 0.000151 | 10/0/0 | −0.001837 ± 0.000192 | 0/10/0 | 10 |
| (3000,4000] | +0.002207 ± 0.000842 | 10/0/0 | −0.001999 ± 0.001031 | 0/10/0 | 10 |
| (4000,5000] | +0.002665 ± 0.001753 | 8/1/0 | −0.002079 ± 0.001771 | 1/8/0 | 9 |

最初の区間から次の区間への**差の変化**は、繁殖参加率差が +0.003406 ± 0.003932（8/10で増加）、死亡率差が −0.002893 ± 0.002792（9/10で低下）。p0の人口shareは全10 seedで増加した。したがって「次の区間で同じ方向」という結果と、「全seedが同じ方向に変化した」という主張は区別する必要がある。

最初の2区間のseed平均をp0/p1/p2で比較すると以下のとおり。各値のSD/min/max/nは `window_comparison.csv` に収録。

| 指標 | p0: (0,1000] | p1: (0,1000] | p2: (0,1000] | p0: (1000,2000] | p1: (1000,2000] | p2: (1000,2000] |
|---|---:|---:|---:|---:|---:|---:|
| population share | 0.003987 | 0.048393 | 0.869943 | 0.095727 | 0.124391 | 0.595077 |
| 繁殖参加率 | 0.013877 | 0.014652 | 0.015262 | 0.016289 | 0.014175 | 0.014267 |
| 死亡率 | 0.007244 | 0.006847 | 0.006497 | 0.005438 | 0.007279 | 0.007584 |
| hunger multiplier | 0.998702 | 0.998675 | 0.998957 | 0.996539 | 0.996849 | 0.996456 |
| thirst multiplier | 1.000242 | 0.999071 | 0.999359 | 0.992319 | 0.993773 | 0.994288 |
| signal probability | 0.101072 | 0.099934 | 0.099746 | 0.099913 | 0.101121 | 0.100217 |
| age | 68.035 | 77.441 | 96.399 | 104.179 | 88.207 | 99.103 |
| generation | 16.758 | 14.839 | 10.848 | 43.960 | 38.930 | 36.113 |

## seed依存・sparse cohort

- (0,1000]のp0繁殖差・死亡差は符号が混在。p0 action exposureは中央値1,595、範囲172〜9,467で、p2の中央値583,532.5、範囲563,568〜614,350より小さい。seed 0のp0は親参加1件・死亡2件のみ。符号多数とseed平均の符号が異なる場合もあり、初期p0率は安定した比較とは限らない。
- (1000,2000]のp0 action exposureは中央値63,635、範囲13,542〜177,200。p2は中央値506,235、範囲394,648〜589,043。全10 seedで計算可能。
- (4000,5000]はseed 8でp2 exposureが0、率・形質差は欠測となりn=9。残りのp2 exposureも1,676〜9,759（中央値3,091）に縮小している。繁殖差が負になるのはseed 3、死亡差が正になるのはseed 1で、両方の方向が一致するのは7/9 seed。
- (1000,2000]の区間平均p0−p2 hunger差は負6/10、thirst差は負7/10、age差は正7/10。形質・年齢の方向は率ほど揃っていない。

## 観測可能な時間構成の交絡

区間内でp0は増加し、p2は減少する。このため各cohortの生存exposureで別々に重み付けすると、**p0平均は後半、p2平均は前半を相対的に強く反映**し得る。世代は時間とともに進むため、世代差をそのまま「同じ時点での世代の違い」と解釈できない。

感度解析として、両cohortが同時に存在するtickだけを使い、同じtickの `mean_p0(t) − mean_p2(t)` を等時間重みで平均した。seed間は引き続き等重みである。これは時間の比較対象を揃えただけで、年齢調整・因果調整ではない。

- (1000,2000]のgeneration差は、別々のexposure重みでは **+7.847 ± 1.596（正10/10）**だが、同じtick比較では **−0.223 ± 0.814（正3/10、負7/10）**。
- (2000,3000]と(3000,4000]では、同じtickでのgeneration差はそれぞれ **−0.854 ± 0.384、−1.210 ± 0.509（いずれも負10/10）**。区間重み付き差の正方向とは逆になる。
- (1000,2000]の同じtick比較ではhunger差 **+0.000534 ± 0.002357（正5/10）**、thirst差 **+0.000067 ± 0.004389（正4/10）**、age差 **+5.381 ± 18.522（正6/10）**。p0の低代謝・高年齢がこの時期に全seed共通だったわけではない。
- (2000,3000]では同じtickでのp0 age差が **+14.966 ± 4.924（正10/10）**。生存者の年齢構成差は観測できるが、それが原因か、低死亡率の結果かは区別できない。

`matched_tick_sensitivity.png` と `matched_tick_difference_summary.csv` に全5区間・5形質の比較を保存した。独立した各cohortの等時間平均による別の感度分析も `equal_tick_weighting_sensitivity.csv` に保存している。後者は両cohortの存在tickを揃えていないため、同じtick比較とは異なる。

## 他形質だけで説明できるか／判定できないこと

「p0は常にhunger/thirst multiplierが低いから率がよい」という単純な説明は支持されない。(1000,2000]はseed 4と9でp0の区間平均hunger/thirstが**両方ともp2以上**なのに、繁殖参加率は高く死亡率は低い。さらに同じtick比較でも代謝形質差の方向は揃っていない。

ただし、**他形質・年齢構成などで説明できないことを証明したわけではない**。現在のcohort平均からは、個体内の形質の組み合わせ、年齢分布、繁殖可能な年齢・energy・cooldownの割合、位置や資源への接近状況、世代と環境の関係を調整できない。learningのcohort指標も存在しない。繁殖参加率には相手側としての参加も含まれ、perceptionの効果、学習の効果、系譜との関連を分離していない。1000tick区間の結果から、優位が生じた正確な時点も特定できない。

追加で確認できたのは、**率の方向の再現性、初期・末期のexposure不足、年齢構成の変化、区間内の時間重み付けによる世代差の反転**である。「perception=0が原因」とは結論しない。

## 出力と再現方法

- `window_metrics_by_seed.csv`：seed×cohort×windowの150行。対象8指標、件数、exposure、有効tick数を保持。
- `window_metrics_summary.csv`：各指標のmean/std/min/max/nとexposure中央値・範囲。
- `window_comparison.csv`：p0/p1/p2を横並びに比較する表。
- `p0_minus_p2_by_seed.csv`、`p0_minus_p2_summary.csv`：対応seed差、分布・符号数・有効n・両cohort exposure。
- `early_transition_by_seed.csv`、`early_transition_summary.csv`：各cohortの最初の2区間の値と変化（第2区間−第1区間）。
- `paired_gap_transition_by_seed.csv`、`paired_gap_transition_summary.csv`：p0−p2差の最初の2区間の変化。
- `equal_tick_weighting_sensitivity.csv`：別々のcohortの等時間平均での差。
- `matched_tick_differences_by_seed.csv`、`matched_tick_difference_summary.csv`：同じtickでの比較。
- `joint_sign_patterns.csv`：率の方向と代謝形質の方向の同時出現数。
- `input_audit.csv`、`analysis_metadata.json`、`validation.json`：記録範囲、入力hash、定義、検証。
- `*_by_window.png`：対象8指標のp0/p1/p2比較、seed実測点、平均±1 SD、n・exposure表。
- `p0_minus_p2_*.png`：対象8指標の対応差、各seedの線、平均±1 SD、符号数、n・exposure表。
- `early_transition.png`：最初の2区間の比較。
- `matched_tick_sensitivity.png`：時間重み付けの感度分析。

```bash
.venv/bin/python -m experiments.analyze_perception_transition
.venv/bin/python -m unittest tests.test_perception_transition -v
```

後処理の合成データ算術テスト8件が成功。実データを標準ライブラリのCSV読み込みと独立計算で検算し、150個のseed/cohort/windowの率・5平均に対する計1,050チェックが成功。既存simulationを実行するテストは今回実行していない。入力CSVのhashは処理前後で不変。同じ入力から再生成したCSV14個・PNG18個・metadataの計33ファイルがバイト単位で一致し、Python/NumPyの乱数状態も不変だった。
