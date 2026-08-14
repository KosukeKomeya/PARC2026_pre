# PARC 2026 — 予選配布環境

PARC 2026予選のための配布環境である。
本リポジトリを用いることで、参加者が実装したポリシーを example タスク上で
実行し、提出前にローカル環境で採点および動作確認を行うことができる。

本環境で実施できる作業は次のとおりである。
- 自身のポリシーを HTTP サーバーとして起動し、Track 1 の example タスクで評価する
- 提出物（zip）を、本番と同一の手順でエンドツーエンドに検証する
- 提出物の妥当性および動作を、提出前に自動チェックする
  （必須ファイルの有無、サーバーが起動して所定の応答を返すこと、
  各リクエストが制限時間内に完了することを確認する）

最初に参照すべきファイルは次のとおりである。
- 提出物の作成方法・動作する最小実装: [submission_template/](submission_template/)
  （`policy_server.py` の `MyPolicy` は編集前でもそのまま動作する）
- 提出前チェック: [validate_submission.py](validate_submission.py)
- 学習の参考例: [examples/](examples/)（提出には必須ではない）
- PARC 2026で使用したπ0.5 LoRA・checkpoint選択・RTC提出の再現手順:
  [examples/pi05_rtc_final_submission.md](examples/pi05_rtc_final_submission.md)

評価パイプラインおよび提出物チェックスクリプトは、本番採点のTrack 1と同じ評価処理・制約を
再現する。ただし、本番評価とは以下の点で異なる。

- 同梱されているのは公開されている example タスクのみである。本番の採点は、
  **公開されていないタスクを含む別のタスクセット**で実施される
- 出力されるのは成功率および軌道メトリクスの生値である。リーダーボードの順位を決定する
  スコア算出設定は含まれない
- 推論タイムアウト（[下記](#タイムアウト仕様)）および成功判定（[下記](#成功判定)）は
  本番と同一である

## 1. セットアップ

Python 3.10、git、unzip が必要である。[setup.sh](setup.sh) は本番の採点環境と同一の構築
（venv、ピン止めした依存、LIBERO-plus の取得とパッチ、アセットのダウンロードと配線）を
一括して実行する。setup.sh が取得・インストールする第三者製ソフトウェアとその
ライセンスは [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照すること。

```bash
bash setup.sh     # 初回のみ（アセット取得を含めて 10〜20 分）
source env.sh     # 評価を実行するシェルで毎回実行する
```

> setup.sh は `~/.libero/config.yaml` を上書きする（既存の設定は `.bak` に退避される）。
> 既に LIBERO を使用しており元の設定に戻す場合は、`~/.libero/config.yaml.bak`
> を書き戻すこと。

### Docker を使用する場合（既存環境への影響を避ける場合はこちらを推奨する）

```bash
docker build -t parc2026 .
docker run -it --rm parc2026                     # 対話シェル（以降のコマンドをそのまま実行できる）
docker run --rm -v $PWD/my_submission.zip:/sub.zip parc2026 \
    python evaluate.py /sub.zip --n-episodes 2   # 提出 zip の一括評価
```

環境構築はローカルと同一の [setup.sh](setup.sh) がビルド時に実行される。

なお、本配布環境の Docker は **CPU 構成**（`ubuntu:22.04` + `torch 2.11.0+cpu`）である。
本番の採点は GPU コンテナ（CUDA 13.0 ベース）で実施されるため、Python の版と評価側
ライブラリのピンは同一であるが、CUDA まわりは異なる。詳細および本番相当（GPU）での
確認方法は[採点環境](#採点環境)を参照すること。

## 2. 評価を回す

```bash
# 1) 自身のポリシーサーバーを起動する（別ターミナル。テンプレートは編集前でも
#    ランダム action を返すので、まずそのまま起動して疎通確認できる）
python submission_template/policy_server.py --port 8000

# 2) 評価を実行する
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2 --max-steps 600

# タスクを指定して評価する（example タスク名を指定する。存在しない名前は候補一覧つきでエラーとなる）
python -m pipeline --server-url http://localhost:8000 --track track1 --tasks <task_id>

# 提出 zip をエンドツーエンドで検証する（zip 展開 → 依存インストール → 評価まで自動実行）
python evaluate.py my_submission.zip --n-episodes 2
```

結果は `results/<submission_id>.json` に出力される。成功率、ステップ数、軌道メトリクス
（経路長、jerk、SPARC 等）の詳細が含まれる。

## 3. 提出前のチェック

提出物の妥当性（必須ファイル、zip 構造、エンドポイント）と、実際に起動して
動作すること（/health→/reset→/act が正常に応答し、応答が制限時間内であること）を検査する。

```bash
python validate_submission.py my_submission.zip            # 静的検査 + 起動スモークテスト
python validate_submission.py my_submission.zip --static   # 静的検査のみ（起動しない）
```

---

## 提出フォーマット

提出物は **HTTP ポリシーサーバー一式の zip** である。サーバーは次の 3 エンドポイントを
実装する（[テンプレート](submission_template/)を編集することで自動的に満たされる）。

| エンドポイント | 役割 |
|---|---|
| `GET /health` | 起動確認（200 を返すまで評価側がポーリングする） |
| `POST /reset` | エピソード開始（`instruction`, `seed` を JSON で受け取る） |
| `POST /act` | 観測（msgpack）→ action を返す。**float32 shape (7,)** `[dx, dy, dz, droll, dpitch, dyaw, gripper]` |

## 採点環境

本番の採点は **GPU コンテナ**上で実施される。構成は次のとおりである。

| 項目 | 値 |
|---|---|
| ベースイメージ | `nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04` |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12（システム Python、`/usr/bin/python3.10`） |
| CUDA Toolkit | **13.0**（`nvcc` V13.0.88。devel ベースのため `nvcc` によるソースビルドが可能） |
| cuDNN | 9.14.0 |
| NCCL | 2.28.3+cuda13.0 |
| PyTorch | **2.11.0+cu130**（`torch.version.cuda` は `13.0`） |
| Triton | 3.6.0 |
| NVIDIA ドライバ | R580 系 |
| レンダリング | `MUJOCO_GL=EGL`（GPU レンダリング） |

評価パイプライン側の主要な依存は次の版で固定されている（提出物からも参照できる）。

```
numpy==1.26.4        mujoco==3.7.0       robosuite==1.4.0    gym==0.25.2      bddl==3.6.0
fastapi==0.140.7     uvicorn==0.51.0     msgpack==1.2.1      requests==2.34.2
huggingface_hub==1.25.1   opencv-python-headless==4.11.0.86  scipy==1.15.3   h5py==3.16.0
Pillow==12.3.0       matplotlib==3.10.9  einops==0.8.2       hydra-core==1.3.2
```

プリインストールされているパッケージの全量は[付録](#付録-採点環境のプリインストール一覧)に示す。

### requirements.txt の書き方

提出物の依存は、**`--system-site-packages` 付きで作成された提出物専用の venv** に対して
`pip install -r requirements.txt` される。したがって次のようになる。

- **`requirements.txt` に書かなかったライブラリは、採点イメージにプリインストール
  されている版がそのまま使われる。** `torch` を書かなければ上表の `2.11.0+cu130` が使われ、
  イメージ側の CUDA 13 のライブラリと整合した状態で GPU が利用できる。
  `numpy` / `fastapi` / `uvicorn` / `msgpack` / `huggingface_hub` 等も同様に省略できる。
- **書いた版は venv 側が優先される。** その版は提出物のサーバーにのみ効き、
  評価パイプライン側には影響しない。

### CUDA 12 系の torch を使用する場合の注意

採点イメージは CUDA 13.0 ベースであり、システム側には **`libnvJitLink.so.13`** しか
存在しない（`.so.12` の代替にはならない）。CUDA 12 ビルドの torch を使用する場合、それが必要とする `nvidia-*-cu12` の
wheel が **venv 側に一式そろっている必要がある**。特に `nvidia-nvjitlink-cu12` が
入らないと、サーバー起動時に次のエラーで失敗する。

```
ImportError: libnvJitLink.so.12: cannot open shared object file: No such file or directory
```

`torch` を単に pin しただけであれば依存として自動的に導入されるが、`requirements.txt` 内に
バージョン衝突がある場合、pip の解決結果が変わって導入されないことがある。
**`pip install` 時に `... requires X, but you have Y` という警告を残さないこと。**
CUDA 12 の wheel 一式が venv にそろっていれば、イメージ側が CUDA 13 でも動作する。

### 提出前の確認

ローカルと本番で torch の構成が異なることに起因する失敗は、次の 1 行で検出できる。
本番相当の環境で、venv 作成 → `pip install -r requirements.txt` の後に実行して
確認することを推奨する。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

本番相当（GPU）を手元で再現する場合は、[Dockerfile](Dockerfile) の `FROM` を
`nvidia/cuda:13.0.3-cudnn-devel-ubuntu22.04` に、[setup.sh](setup.sh) の torch を
`torch==2.11.0`（`--index-url https://download.pytorch.org/whl/cu130`）に差し替えて
ビルドし、`docker run --gpus all` で起動すること。

## 成功判定

本番の採点と同一の基準である。エピソードが成功と扱われるのは、**タスクのゴール条件を
満たし、かつ衝突が発生していない**場合のみである。

衝突は「操作対象以外の物体を動かしたか」で判定する。タスクが操作対象とする物体
（BDDL の `:obj_of_interest`）を除く全物体について、初期位置からの変位（xyz 各軸の
絶対値の和）を各ステップで監視し、その最大値が **1 mm** を超えた物体が 1 つでもあれば、
そのエピソードは失敗となる。

- 対象物体を掴んで動かすことは当然に許容される。判定対象は「それ以外の物体」である
- 変位は環境が落ち着いた時点（エピソード開始直前）の位置を基準とする
- 動かしてしまった物体を元の位置へ戻しても、変位の最大値で判定するため失敗のままである

## タイムアウト仕様

本番のTrack 1採点と同一の制約である。

**`/act`・`/reset` の 1 リクエストが 10 秒を超えた場合、そのトラックは失敗（error 扱い）
となり 0 点となる。** これは平均でも累積でもなく、1 回でも超過するとそのトラック全体が
失敗となる制約である。モデルの推論が 10 秒以内に収まることを必ず確認すること。

| 対象 | 上限 | 超えると |
|---|---|---|
| 推論: `/act`（および `/reset`）1 リクエスト | **10 秒** | そのトラックは error 扱いの 0 点 |
| サーバー起動（モデルロードを含む） | 既定 **120 秒**（`SERVER_TIMEOUT` で変更可） | 評価不能として終了 |

- タイムアウトは **HTTP リクエスト単位**である。平均・累積・エピソード単位の制限はない。
- アクションチャンクをサーバー内にキャッシュするモデルの場合、推論が実行される「重い」
  リクエストのみが上限の対象となる（実質的な制約は「チャンク 1 回分の推論 ≤ 10 秒」である）。
- [validate_submission.py](validate_submission.py) のスモークテストは、同一の 10 秒基準で
  レイテンシを警告する。提出前に必ず一度実行することを推奨する。

## ディレクトリ構成

| パス | 役割 |
|---|---|
| [pipeline/](pipeline/) | Track 1 評価パイプライン |
| [compe/t1/](compe/t1/) | Track 1 の example タスク定義 |
| [submission_template/](submission_template/) | 提出テンプレート（`policy_server.py` の `MyPolicy` のみ編集。編集前でも動作する） |
| [evaluate.py](evaluate.py) | Track 1 の zip 一括評価 |
| [validate_submission.py](validate_submission.py) | 提出物チェックスクリプト |
| [examples/](examples/) | 学習の参考例（SmolVLA の LoRA 追加学習ノートブック）。提出には必須ではない |
| [tests/](tests/) | ハーネスの単体テスト |
| [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) | setup.sh が取得する第三者製ソフトのライセンス表記 |

## 付録: 採点環境のプリインストール一覧

本番採点イメージのシステム Python 3.10 における `pip list` である。
`requirements.txt` に書かなかったライブラリは、ここに記載の版が使用される。

なお、この一覧は評価パイプライン側の依存であり、**GPU イメージにはこれに加えて
ベースライン実装（pi0.5 / openpi JAX）の依存が焼き込まれている**
（`jax[cuda12]==0.5.3` / `jaxlib==0.5.3` / `flax==0.10.2` / `orbax-checkpoint==0.11.13` /
`transformers==4.53.2` およびそれらが引く `nvidia-*-cu12` の wheel 等）。
イメージ更新に伴って変わりうるため、**自分のモデルに必要な依存は
`requirements.txt` に明示すること**（イメージ側に入っていることを前提にしない）。

<details>
<summary>pip list（全量）</summary>

```
absl-py==2.5.0                      annotated-doc==0.0.4                annotated-types==0.8.0
antlr4-python3-runtime==4.9.3       anyio==4.14.2                       attrs==26.1.0
bddl==3.6.0                         certifi==2026.7.22                  charset-normalizer==3.4.9
click==8.4.2                        cloudpickle==3.1.2                  contourpy==1.3.2
cuda-bindings==13.3.1               cuda-pathfinder==1.6.0              cuda-toolkit==13.0.2
cycler==0.12.1                      defusedxml==0.7.1                   easydict==1.13
einops==0.8.2                       etils==1.13.0                       exceptiongroup==1.3.1
fastapi==0.140.7                    fastjsonschema==2.22.1              filelock==3.32.0
fonttools==4.63.0                   fsspec==2026.6.0                    future==1.0.0
glfw==2.10.2                        gym==0.25.2                         gym-notices==0.1.0
h11==0.16.0                         h5py==3.16.0                        hf-xet==1.5.2
httpcore==1.0.9                     httpx==0.28.1                       huggingface_hub==1.25.1
hydra-core==1.3.2                   idna==3.18                          ImageIO==2.37.4
importlib_resources==7.1.0          iniconfig==2.3.0                    Jinja2==3.1.6
joblib==1.5.3                       jsonschema==4.26.0                  jsonschema-specifications==2025.9.1
jupyter_core==5.9.1                 jupytext==1.19.5                    kiwisolver==1.5.0
lazy-loader==0.5                    llvmlite==0.48.0                    markdown-it-py==4.2.0
MarkupSafe==3.0.3                   matplotlib==3.10.9                  mdit-py-plugins==0.6.1
mdurl==0.1.2                        mpmath==1.3.0                       msgpack==1.2.1
mujoco==3.7.0                       nbformat==5.10.4                    networkx==3.4.2
nltk==3.10.0                        numba==0.66.0                       numpy==1.26.4
nvidia-cublas==13.1.0.3             nvidia-cuda-cupti==13.0.85          nvidia-cuda-nvrtc==13.0.88
nvidia-cuda-runtime==13.0.96        nvidia-cudnn-cu13==9.19.0.56        nvidia-cufft==12.0.0.61
nvidia-cufile==1.15.1.6             nvidia-curand==10.4.0.35            nvidia-cusolver==12.0.4.66
nvidia-cusparse==12.6.3.3           nvidia-cusparselt-cu13==0.8.0       nvidia-nccl-cu13==2.28.9
nvidia-nvjitlink==13.0.88           nvidia-nvshmem-cu13==3.4.5          nvidia-nvtx==13.0.85
omegaconf==2.3.1                    opencv-python==4.11.0.86            opencv-python-headless==4.11.0.86
packaging==26.2                     pillow==12.3.0                      pip==26.1.2
platformdirs==4.11.0                pluggy==1.6.0                       pydantic==2.13.4
pydantic_core==2.46.4               Pygments==2.20.0                    PyOpenGL==3.1.10
pyparsing==3.3.2                    pytest==9.1.1                       python-dateutil==2.9.0.post0
PyYAML==6.0.3                       referencing==0.37.0                 regex==2026.7.19
requests==2.34.2                    robosuite==1.4.0                    rpds-py==0.30.0
scikit-image==0.25.2                scipy==1.15.3                       setuptools==81.0.0
six==1.17.0                         starlette==1.3.1                    sympy==1.14.0
termcolor==3.3.0                    tifffile==2025.5.10                 tomli==2.4.1
torch==2.11.0+cu130                 tqdm==4.70.0                        traitlets==5.15.1
triton==3.6.0                       typing_extensions==4.16.0           typing-inspection==0.4.2
urllib3==2.7.0                      uvicorn==0.51.0                     Wand==0.7.2
wheel==0.47.0                       zipp==4.1.0
```

</details>

> 上記の `nvidia-*` は **CUDA 13 系**（`-cu13` または版番号 13.x）である。CUDA 12 ビルドの
> torch を使用する場合は、必要な `nvidia-*-cu12` を `requirements.txt` に自分で含めること
> （[CUDA 12 系の torch を使用する場合の注意](#cuda-12-系の-torch-を使用する場合の注意)）。
