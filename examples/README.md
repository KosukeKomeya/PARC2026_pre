# 参考例（examples）

## 最終π0.5ワークフロー

PARC 2026 Track 1で使用したπ0.5の流れは、役割ごとに次の3本へ分けている。

| 段階 | ファイル | 役割 |
|---|---|---|
| 学習 | [pi05_qkvo_experiment_colab.ipynb](pi05_qkvo_experiment_colab.ipynb) | Action Expertのq/k/v/oへLoRA追加学習し、Driveへcheckpointを保存する |
| 選択・診断 | [pi05_checkpoint_sweep_colab.ipynb](pi05_checkpoint_sweep_colab.ipynb) | checkpoint 1500/2000/2500/3000を同一条件で比較し、step 2500を選択する |
| 最終評価・提出 | [pi05_rtc_final_submission.md](pi05_rtc_final_submission.md) | Replan 10、Inference 8、RTC horizon 10 / guidance 5.0で評価し、検証済みZIPと再現manifestを作る |

共通処理は`pi05_action_expert_lora.py`、`pi05_checkpoint_sweep.py`、
`pi05_finalize_qkvo_colab.py`へ分離している。提出時の実装は
`submission_template/policy_server.py`であり、モデル重み・提出ZIP・動画・評価結果は
Gitへコミットしない。

## ベースラインと参考実験

| ファイル | 内容 |
|---|---|
| [smolvla_libero_spatial_lora.ipynb](smolvla_libero_spatial_lora.ipynb) | SmolVLA を LIBERO-plus Spatial で LoRA 追加学習する Google Colab ノートブック |
| [pi05_parc_colab.ipynb](pi05_parc_colab.ipynb) | π0.5-LIBERO導入とq/v LoRAのベースラインを確認するGoogle Colabノートブック |

## pi05_qkvo_experiment_colab.ipynb

`pi05_qkvo_experiment_colab.ipynb` is an isolated follow-up experiment. It
keeps the accepted q/v submission untouched and changes only the Action Expert
Attention LoRA targets to q/k/v/o. Run it in a fresh Colab runtime so the
comparison starts from the original public pi0.5 checkpoint. Checkpoints and a
winning merged model are persisted under a separate Google Drive directory.

## pi05_parc_colab.ipynb

`lerobot/pi05_libero_finetuned_v044` をPARCの `MyPolicy` に接続するための
導入・提出物作成ノートブックである。固定コミットのLeRobot/PyTorch実装と
patched Transformers、固定revisionの重みを使い、次を実行する。

1. Python 3.10の分離環境を準備する
2. 実モデルをロードし、128×128の2カメラ入力から7次元actionを推論する
3. 重い推論が10秒未満か確認する
4. 採点時に外部通信を行わない `pi05_submission.zip` を作る
5. 提出バリデータの静的検査を実行する
6. PARC配布キットの公開4タスクをHTTP経由で評価し、リアルタイム進捗と
   失敗時だけ保存するagent/wrist 2画面の評価動画を確認する

PaliGemma tokenizerの利用条件への同意とHugging Faceへのログインが必要である。
π0.5は4B規模なので、ColabではL4またはA100を推奨する。T4ではメモリ不足や
10秒制限超過の可能性がある。

このノートブックは既存のLIBERO fine-tuned checkpointの**導入用**であり、
追加学習は行わない。LIBERO-plusで追加学習する場合は、公式設定でも80GB GPUを
前提とするため、まずこの版で提出パイプラインと評価を成立させてから別実験にする。

## smolvla_libero_spatial_lora.ipynb

`lerobot/smolvla_libero_plus` を初期重みとし、LIBERO-plus Spatial の 10 タスクを
LoRA で追加学習する。学習後は LoRA を元の重みへマージし、追加学習の前後を
同一条件で比較する。

### 使い方

1. Google Colab で開き、ランタイムのタイプを GPU（T4 で足りる）に変更する
2. 上から順に実行する。所要時間は T4 で数時間程度である
3. マージ済みモデル一式（zip）と、追加学習前後の成功率の比較（CSV）が出力される

学習条件は 10 タスク × 各 5 エピソード（計 50 エピソード）、3,000 steps、
バッチサイズ 1 で、Colab で完走することを優先した最小構成である。
性能を伸ばす場合はここを出発点に、自身の環境で条件を組み直すとよい。

### 提出物にするまでの作業

出力されるのは LeRobot 形式のモデル重みであり、これ単体では提出できない。
[submission_template/](../submission_template/) の `MyPolicy` にモデルを組み込み、
ポリシーサーバーの形にする。観測と action の仕様は
[submission_template/policy_server.py](../submission_template/policy_server.py)
の docstring にある。

推論は 1 リクエストあたり 10 秒以内に収める必要がある
（[ルートの README](../README.md#タイムアウト仕様)）。

### ノートブック内の評価と、本番の採点の違い

ノートブック内の評価は学習の効果を手早く確認するためのもので、採点とは条件が異なる。
出てくる成功率は本番スコアの目安にはならない。

| 項目 | ノートブック | 本番の採点 |
|---|---|---|
| 評価タスク | LIBERO-plus Spatial の 10 タスク | Track 1（`compe/t1/` のタスクセット） |
| 実行方法 | LeRobot の `lerobot-eval` | `python -m pipeline` + 提出したポリシーサーバー |
| 観測の解像度 | 256×256 | 128×128 |
| 1 タスクあたりの試行数 | 3（`EVAL_EPISODES_PER_TASK` で変更可） | 非公開（配布キットの既定は 20） |

試行数が 3 のままだと 1 エピソードの成否で成功率が約 33 ポイント動くため、
追加学習の前後を比べる場合は `EVAL_EPISODES_PER_TASK` を増やすこと。

### 実行環境

ノートブックの環境構築は Colab 向けで、[setup.sh](../setup.sh) とは独立している。
依存パッケージのバージョンが一致しない箇所があるため、評価と提出前チェックは
リポジトリ側の環境（`setup.sh` + `env.sh`）で行うこと。

ノートブックが利用する第三者製ソフトウェア・モデル・データセットのライセンスは、
各配布元の表記を参照すること。
