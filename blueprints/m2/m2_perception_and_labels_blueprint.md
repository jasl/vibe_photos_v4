# M2 Perception & Labeling Blueprint

Status: ready for implementation design review (focuses on perception quality, label layer, and clustering on top of the existing M1 pipeline)

> 本文是和 GPT 反复讨论后沉淀下来的 M2 方案整理版，用来指导后续实现与评估。

---

## 1. Background and TL;DR

M1 已经完成了一条稳定的本地预处理流水线：

- **感知输出（features）**：SigLIP 全图 embedding、BLIP caption、OWL‑ViT region 检测、pHash。
- **轻量分类**：基于 SigLIP 的 coarse scene + 布尔属性（has_text / has_person / is_document / is_screenshot）。
- **Cache + primary DB**：感知结果写入主数据库（M1 时代为 `data/index.db`，现已固定为 PostgreSQL + pgvector；`data/index.db` 仅作历史背景），缓存文件位于 `cache/` 目录。
- **Flask QA UI**：支持按 coarse scene 和布尔属性筛图。

在此基础上，**M2 的核心目标**是：

1. **彻底解耦「特征」和「标签」**：SigLIP / BLIP / OWL‑ViT 只当「特征工厂」，输出 embedding、caption、bbox，不再直接生成最终标签。
2. **重构统一标签层（label layer）**：用 `labels + label_aliases + label_assignment` 三张表承载所有 scene / object / cluster / 属性标签，记录来源和版本，支持重算。
3. **显著提高感知质量**：
   - 一级 scene + 布尔属性更多依赖「规则 + 专用 detector + 小分类器」，减少 prompt‑pair 二分类依赖。
   - Object labels 采用 embedding + prototype 的 zero‑shot / few‑shot 策略，而不是配置里写死 label 列表。
4. **支持「完全不认识的新东西，但能把相似的照片组织在一起」**：
   - 对整图 embedding 和 region embedding 做聚类，形成「相似物体簇」并映射为标签。
5. **为后续 M4 Learning / 个性化铺好地基**：
   - embedding 带版本，`label_assignment` 支持多 source；
   - 聚类结果 + 人工修正可以直接作为未来 few‑shot 训练数据。

一句话总结：

> **M2 = 统一标签层（label_assignment） + Region SigLIP 特征 + Object zero‑shot 标签 + 聚类兜底，Scene 先沿用现有逻辑，线性头与个性化放到后续迭代。**

---

## 2. Scope and Non‑Goals (M2)

### 2.1 In Scope

- **存储与基础设施**
  - 继续使用现有的 **PostgreSQL + 文件系统 cache** 结构（早期文件数据库方案仅作历史背景，不再作为运行时目标）。
  - 不大改 Celery / pipeline 基础框架，只在现有阶段上重排职责、加新任务。

- **感知质量目标**
  - 在复用现有 scene classifier 的前提下，通过规则特征与标签层重构，让一级 scene 分类在 QA 场景下「更可控、易评估」，整体达到「基本靠谱」水平：
    - `LANDSCAPE / PEOPLE / FOOD / PRODUCT / DOCUMENT / SCREENSHOT / OTHER` 等能明显区分，大规模精度提升留给后续线性头。
  - 布尔属性质量显著优于 M1：
    - `has_person / has_text / is_document / is_screenshot`，容错性更好、更稳定。
  - 常见物体（电脑、手机、耳机、饮品、常见食物）的 object labels 在 QA 中「可信度可用」。
  - 对于新产品 / 冷门设备：
    - 即便模型认不出具体名称，也能将相似图片/region 聚成簇，UI 一眼看出「这一堆是同一件东西」。

- **统一标签层**
  - 引入 `labels / label_aliases / label_assignment` 三表，承载：
    - scene labels（一级场景）
    - attribute labels（布尔属性）
    - object labels（可识别物体）
    - cluster labels（相似物体簇）
  - 所有标签带：
    - `level`（scene / object / cluster / attribute）
    - `target_type`（image / region）
    - `source`（rule / zero_shot / classifier / cluster / manual / aggregated）
    - `label_space_ver`（scene_v1 / object_v1 / cluster_v1 等）

- **聚类与相似物体簇**
  - 对 **scene=PRODUCT/FOOD** 的整图 embedding 做简单 kNN + 连通分量聚类。
  - 对 region embedding 做类似聚类，得到细粒度物体簇。
  - 每个 cluster 同步为一个 cluster label，并给成员写 `label_assignment`。

- **评估与工具**
  - 定义小规模标注集（约 800–1500 张图）。
  - 实现简单的评估脚本，计算 scene 准确率、布尔属性 precision/recall、object label top‑k 命中率。

### 2.2 Out of Scope

- 不在 M2：
  - 不引入 PostgreSQL / pgvector / docker‑compose 一体化部署（属于 M3）。
  - 不做完整「标注 UI + 交互式批量编辑」体验（属于 M4）。
  - 不做大规模本地 fine‑tune，仅预留接口和数据形态。
  - 不追求最终形态的搜索体验（M3 再优化）。

---

## 3. Architecture Overview (M2 View)

### 3.1 Three‑Layer View

M2 把系统划分为三个清晰层次：

1. **Perception / Features（感知层）**
   - 已有：
     - **SigLIP image embedding**（全图）
     - **BLIP caption**
     - **OWL‑ViT region detection**（bbox + raw label + score）
     - **SigLIP region embedding**（新增，基于 bbox 裁剪）
     - **pHash**（perceptual hash）
   - 新增 & 强调：
     - 可选 `face_detector`（只输出 has_face / face_count，尽量轻量）
     - 用规则特征推断 screenshot / document 倾向（长宽比、EXIF 有无、亮度、色块等）
   - **职责**：只负责输出「无语义」数值特征（embedding）+ 几何信息（bbox）+ 原始检测结果，不直接做标签决策。

2. **Label Layer（标签层）**
   - 输入：感知层输出 + Label schema + 规则/小模型结果。
   - 输出：写入统一的 `label_assignment` 表：
     - 一级 scene labels（`scene.*`）
     - 布尔属性 labels（`attr.*`）
     - Object labels（`object.*`）
     - Cluster labels（`cluster.*`）
   - 每条标签记录：
     - `target_type`：image / region
     - `target_id`：image_id 或 region_id
     - `label_id`：指向 `labels` 表
     - `source`：rule / zero_shot / classifier / cluster / manual / aggregated
     - `label_space_ver`：scene_v1 / object_v1 / cluster_v1 …
     - `score`：相似度 / 概率 / 置信度
     - `extra_json`：如 margin、top_k、classifier_version 等。
   - 现有的 scene / has_person 等字段逐步「镜像」到 label 层，后续 UI 和检索优先对 label 层聚合。

3. **Retrieval & UI Layer（检索与 UI 层）**
  - 继续使用当前 Flask QA UI + PostgreSQL，但改成：
     - 一级分类 facet 基于 `label_assignment` + `labels`（`level='scene'`）聚合。
     - Region 标签从 `label_assignment` 读取（`target_type='region'`）。
     - Cluster 当作特殊标签展示（相似物体组），支持按 cluster 浏览。
   - 默认 UI 行为建议（M2 视角）：
     - 列表页：
       - 左侧 facet：scene（`scene.*`）、object（`object.*`）、cluster（`cluster.*`）、属性（`attr.*`）；
       - 顶部搜索框：自然语言 → caption FTS + image embedding 召回，再用上述标签做过滤。
     - 单图页：
       - 展示：scene/attribute/object/cluster 标签，以及 region 框可视化；
       - 后续 M4 可在此基础上增加：修改 object 标签、cluster 命名、批量应用到相似图片等交互。
   - 后续 M3/M4 的 API / Web UI 可以直接基于 label 层演化，不再改动底层特征抽取。

### 3.2 Pipeline Flow Reordering

当前 M1 的主流程大致为：

```text
扫描 → hash → phash → SigLIP embedding → BLIP caption
    → coarse scene + attributes (zero-shot)
    → (可选) detection + SigLIP region re-ranking
    → 写 PostgreSQL
```

M2 调整为三阶段：

1. **Preprocessing Stage（预处理阶段）**  
   （保持 M1，必要时小幅整理）
   - 扫描目录，计算 content hash（image_id）与元数据。
   - 计算 pHash + near‑duplicate 分组。
   - 生成缩略图等。

2. **Feature Stage（特征阶段，Feature Pass）**
   - 全图 SigLIP embedding。
   - BLIP caption。
   - OWL‑ViT detection + bbox（regions）。
   - 对每个 region 裁剪 patch，计算 SigLIP region embedding。
   - （可选）face detection / screenshot/doc 规则特征。
   - 所有输出视为「model features」，只在 cache 和投影表中记录，省略最终标签决策。

3. **Label Stage（标签阶段，Label Pass）→ 可多次重跑**
   - Scene label pass：
     - 基于 image embedding + 规则特征（包含人脸 / 文本等）生成 scene + attrs 标签。
   - Object label pass：
     - 基于 region embedding + 文本 prototype（label_alias SigLIP embedding）生成 object 标签。
   - Cluster pass：
     - 基于 image / region embedding 做聚类，生成 cluster 标签。

**重要约束**：

- 特征阶段只依赖模型权重版本及 cache manifest；**仅在模型升级时重跑**。
- 标签阶段完全解耦，可根据 `label_space_ver` 与规则更新随时重算。

---

## 4. Data Model Changes

在不破坏已有表的前提下，M2 需要为标签层和 region/cluster 增加若干表。此处以 PostgreSQL schema 为准，未来可直接扩展至 pgvector。

### 4.1 Label Schema

#### 4.1.1 `labels`

统一管理所有标签（scene / object / attribute / cluster）。示意：

```sql
CREATE TABLE labels (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  key               TEXT NOT NULL UNIQUE,  -- "scene.food" / "object.macbook" / "cluster.siglip_image_knn_v1.17"
  display_name      TEXT NOT NULL,         -- UI 默认显示名（可中文）
  level             TEXT NOT NULL,         -- 'scene' | 'object' | 'cluster' | 'attribute'
  parent_id         INTEGER NULL,          -- 父标签（用于层级结构）
  icon              TEXT NULL,             -- 可选：UI icon 名称
  is_active         INTEGER NOT NULL DEFAULT 1,
  created_at        REAL NOT NULL,
  updated_at        REAL NOT NULL,
  FOREIGN KEY(parent_id) REFERENCES labels(id)
);
```

标签树示例（部分）：

- `scene.food`（顶层）
  - `scene.food.chinese`
  - `scene.food.western`
- `object.electronics`（物体类）
  - `object.electronics.laptop`
    - `object.electronics.laptop.macbook`
  - `object.electronics.phone`

#### 4.1.2 `label_aliases`

支持多语言、多写法的别名，用于 SigLIP 文本 embedding。

```sql
CREATE TABLE label_aliases (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  label_id      INTEGER NOT NULL,
  alias_text    TEXT NOT NULL,
  language      TEXT NULL,   -- 'en' | 'zh' | NULL
  is_preferred  INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(label_id) REFERENCES labels(id)
);
```

示例：

- `object.electronics.laptop.macbook`：
  - `"macbook"`, `"apple macbook laptop"`, `"a silver macbook pro on a desk"`
- `object.food.pizza`：
  - `"pizza"`, `"slice of pizza on a plate"`, `"a whole pizza on a plate"`

这些 alias 文本会统一用 SigLIP 文本 encoder 编成 embedding，再平均得到 label prototype。

#### 4.1.3 `label_assignment`

所有标签赋值统一落在此表：

```sql
CREATE TABLE label_assignment (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  target_type        TEXT NOT NULL,   -- 'image' | 'region'
  target_id          TEXT NOT NULL,   -- image_id 或 region_id
  label_id           INTEGER NOT NULL,
  source             TEXT NOT NULL,   -- 'rule' | 'classifier' | 'zero_shot' | 'cluster' | 'manual' | 'aggregated'
  label_space_ver    TEXT NOT NULL,   -- e.g. 'scene_v1', 'object_v1'
  score              REAL NOT NULL,   -- 相似度/概率/置信度
  extra_json         TEXT NULL,       -- { "margin": 0.12, "top_k": 3, ... }
  created_at         REAL NOT NULL,
  updated_at         REAL NOT NULL,
  UNIQUE(target_type, target_id, label_id, source, label_space_ver),
  FOREIGN KEY(label_id) REFERENCES labels(id)
);
```

使用约定：

- `source='classifier'`：来自 scene classifier 或小分类头。
- `source='rule'`：来自规则或简单打分（例如 screenshot/doc 规则）。
- `source='zero_shot'`：SigLIP zero‑shot region/object 标签。
- `source='aggregated'`：由 region 聚合到 image 的标签。
- `source='cluster'`：聚类得到的 cluster label。
- `source='manual'`：人工标注或 UI 批量操作。

时间戳与分数约定：

- `created_at` / `updated_at` 统一使用 Unix 时间戳（毫秒，REAL）。
- `label_assignment.score` 统一用 0–1 的概率/相似度刻度：
  - classifier / rule / zero_shot / aggregated：概率或余弦相似度；
  - cluster：固定 1.0，真实距离存放在 `cluster_membership.distance`；
  - manual / duplicate_propagated：固定 1.0（表示肯定的标签）。

#### 4.1.4 `label_space_ver`：标签空间版本管理

为了支持「同一张图片可以被不同版本的算法多次打标签」而不互相踩踏，M2 引入 `label_space_ver` 概念，对应一套完整的标签空间与算法配置，例如：

- `scene_v1`：当前使用的场景分类器 + 属性规则（SceneClassifierWithAttributes + 规则）
- `scene_v2`：未来基于线性头 / fine-tune 模型的新版场景分类
- `object_v1`：当前 object zero-shot 策略（SigLIP 文本原型 + region embedding）
- `cluster_v1`：当前 kNN + 连通分量聚类策略

约定：

- **每一条 `label_assignment` 必须带上 `label_space_ver`**，表示它属于哪一版算法。
- 当算法 / 标签体系有明显变化（例如改了标签枚举、阈值策略、换模型），**用新版本号**：
  - 原有记录保留，方便对比、回滚；
  - 新版本写入 `scene_v2` / `object_v2` 等。
- 默认情况下：
  - 代码侧有常量集：`scene_v1 / object_v1 / cluster_v1`；
  - 配置侧（如 `label_spaces.scene_current: scene_v1`）指定当前 active 版本，用于写入与查询；
  - UI 与检索按单一 active 版本运行，评估脚本可同时读取多版本（`scene_v1` vs `scene_v2`）做对比报告。
- 不同来源可以共存在同一版本空间内，比如：
  - `source='classifier'`：自动模型输出；
  - `source='manual'`：人工修正，UI 查询时可优先使用人工标签覆盖模型标签。

一个典型的配置示意为：

```yaml
label_spaces:
  scene_current: scene_v1
  object_current: object_v1
  cluster_current: cluster_v1
```

这样，**标签层可以安全地持续演进**，而不会出现“重算一次就把历史结果全覆盖”的问题。

进一步约定 `label_space_ver` 的语义与使用方式：

- 对于某一类标签（scene / object / cluster），同一 `label_space_ver` 表示「在一套固定模型 + 超参配置下，对全库 target 的**完整标签视图**」。
- 若仅做 bugfix 或轻微阈值调整且不改变整体语义，可在同一 `label_space_ver` 上重跑 label pass：依赖 `UNIQUE(target_type, target_id, label_id, source, label_space_ver)` 约束，重跑会覆盖 `score / extra_json / updated_at`，语义上视为**全量刷新**，无需新起版本号。
- 当发生以下任意情况时，应 bump 新版本号（如 `scene_v2` / `object_v2` / `cluster_v2`）：
  - 更换底层模型 checkpoint 或 embedding 维度；
  - 标签集合发生明显变化（增加/删除 scene / object / attribute / cluster key）；
  - 打分逻辑或决策边界（score → label 的映射）有实质性调整，导致新旧结果不可直接对比。
- 清理策略建议：
  - 线上环境通常只保留每类标签最近 1–2 个版本，其余通过 CLI 显式清理；
  - 清理某版本可直接删除对应 `label_assignment` 记录，例如：

```sql
DELETE FROM label_assignment WHERE label_space_ver = 'scene_v0';
```

- UI 与检索始终只使用配置中声明的当前版本（如 `label_spaces.scene_current`），评估脚本可以显式指定多个 `label_space_ver` 并行读取做对比。

### 4.2 Region & Region Embedding

为避免在标签层反查 JSON，建议在主库中增加 region 与 region embedding 的简化表：

```sql
CREATE TABLE regions (
  id             TEXT PRIMARY KEY,   -- e.g. "{image_id}#{idx}"
  image_id       TEXT NOT NULL,
  x_min          REAL NOT NULL,
  y_min          REAL NOT NULL,
  x_max          REAL NOT NULL,
  y_max          REAL NOT NULL,
  detector       TEXT NOT NULL,      -- 'owlvit-base'
  raw_label      TEXT NULL,          -- OWL-ViT 原始 label，可选
  raw_score      REAL NOT NULL,
  created_at     REAL NOT NULL
);

CREATE TABLE region_embedding (
  region_id      TEXT NOT NULL,
  model_name     TEXT NOT NULL,
  embedding_path TEXT NOT NULL,
  embedding_dim  INTEGER NOT NULL,
  backend        TEXT NOT NULL,
  updated_at     REAL NOT NULL,
  PRIMARY KEY (region_id, model_name),
  FOREIGN KEY(region_id) REFERENCES regions(id)
);
```

命名与关联约定：

- `images.image_id` 始终等于内容哈希，不因 canonical 选择或传播行为增加后缀。
- `regions.id` 统一使用 `{image_id}#{index}`；`label_assignment.target_id` 对 `image` / `region` 分别对应 `image_id` / `regions.id`。
- near-duplicate 的传播使用独立来源（如 `source='duplicate_propagated'`），不复用同一 `image_id`。

#### 4.2.1 Detection 阶段输出内容约定（只产特征，不产语义标签）

从 M2 开始，检测阶段被明确归类为「感知层」，**只负责产出几何信息与 embedding，不再承担“命名物体”的职责**，避免 detection 阶段和标签层耦合导致的重算成本和复杂度。

约定 detection 阶段输出：

1. **`regions` 表（必须）**

   - 每个检测框一条记录，字段包含：
     - `id`: region 唯一 ID（如 `"{image_id}#{index}"`）
     - `image_id`: 所属图片
     - `x_min, y_min, x_max, y_max`: 归一化坐标（相对宽高）
     - `detector`: 使用的检测模型（如 `"owlvit-base"`）
     - `raw_label`: 检测模型原始输出的类别名（可选，仅作 debug 参考）
     - `raw_score`: 模型原始置信度分数
     - `created_at`: 时间戳

   > 注意：这里的 `raw_label` 不视为系统正式标签，只是模型原始 insight，**不会直接暴露给 UI** 或被用作检索条件。

2. **`region_embedding` 表（必须）**

   - 每个 region 的 SigLIP embedding 一条记录，字段包含：
     - `region_id`: 对应 `regions.id`
     - `model_name`: 使用的 embedding 模型（如 `siglip-base-patch16-224`）
     - `embedding_path`: `.npy` 文件相对路径
     - `embedding_dim`: 向量维度
     - `backend`: 推理后端（PyTorch / ONNX 等）
     - `updated_at`: 时间戳

   - 文件内容为 L2-normalized 的向量，可直接用于：
     - object zero-shot label pass；
     - region-level clustering；
     - 未来的 few-shot 分类头 / 细粒度识别。
   - `embedding_path` 以 `cache_root/embeddings/` 为基准，命名示例：
     - image：`images/{model_name}/{image_id}.npy`
     - region：`regions/{model_name}/{region_id}.npy`
   - `model_name` 带版本后缀（如 `siglip-base-patch16-224:v1`），换模型或量化方式时更名；`.npy` 一律保存 L2-normalized 的 float32 向量，下游可直接点积做 cosine。

3. **可选 JSON Cache（仅用于 debug / QA）**

   - 如 `{cache_root}/detections/{image_id}.json`，记录：
     - 原始 detector 输出（logits / 置信度 / 过滤前的框）；
     - 内部调试信息（NMS 中间结果等）。
   - JSON cache 是 **内部调试资产**，不作为系统持久语义的一部分，也不会直接影响标签层。

4. **Detection 阶段不做的事情（明确禁止）**

   - 不再：
     - 计算 region 对文本 label 的相似度；
     - 选择最终物体标签名；
     - 写入 `label_assignment` 表；
     - 对 object label / cluster 做任何决策。
   - 所有语义相关的逻辑（object label、cluster label、场景判断等）**统一通过标签层的各种 pass 来完成**（object label pass / scene label pass / cluster pass 等）。

这一约定保证了：

- detection 阶段可以独立演进（换模型 / 改阈值），只影响 `regions + region_embedding`；
- 标签层可以自由试验不同策略，而不需要每次都重跑 heavy detection；
- 整个 pipeline 的职责划分清晰：
  - detection 只负责“看到什么”；
  - label passes 决定“怎么称呼它 / 怎么分组”。

### 4.3 Clusters & Membership (Optional but Recommended)

可以用专门的 cluster 表，也可以完全复用 `labels(level='cluster')`。为了便于后续算法升级和记录参数，推荐单独存结构：

```sql
CREATE TABLE image_similarity_cluster (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  key            TEXT NOT NULL UNIQUE,   -- 'cluster.siglip_image_knn_v1.17'
  method         TEXT NOT NULL,          -- 'siglip_image_knn_v1' | 'siglip_region_knn_v1' 等
  params_json    TEXT NOT NULL,          -- 算法参数 JSON
  created_at     REAL NOT NULL
);

CREATE TABLE cluster_membership (
  cluster_id     INTEGER NOT NULL,
  target_type    TEXT NOT NULL,          -- 'image' | 'region'
  target_id      TEXT NOT NULL,
  distance       REAL NOT NULL,          -- 到 cluster 原型或邻居的距离
  is_center      INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(cluster_id, target_type, target_id),
  FOREIGN KEY(cluster_id) REFERENCES image_similarity_cluster(id)
);
```

标签层可以简单地把每个 cluster 映射成一个 `labels` 记录（`level='cluster'`），并将簇成员写入 `label_assignment(source='cluster')`。

为了避免多次重跑同一聚类算法导致结果混乱，约定 cluster 的生命周期与重跑策略如下：

- 一个聚类运行由 `(method, label_space_ver)` 唯一标识：
  - `image_similarity_cluster.method` 标记算法与配置（如 `siglip_image_knn_v1` / `siglip_region_knn_v1`）；
  - `label_space_ver`（如 `cluster_v1`）标记写入的标签空间版本，`params_json` 记录具体参数（k、sim_threshold 等）。
- 创建 cluster 时：
  - `image_similarity_cluster.id` 为自增整数；
  - 对每条新建记录设置 `key = "cluster.{method}.{id}"`；
  - 在 `labels` 中同时插入一条 `level='cluster'`、`key` 相同的记录，生命周期与对应 cluster 完全一致。
- 对同一 `(method, label_space_ver)` 重跑时，视为**全量刷新**：
  - 先删除该组合下所有旧结果：
    - `cluster_membership` 中的成员关系；
    - `image_similarity_cluster` 中对应记录；
    - `labels(level='cluster')` 中对应 `key`；
    - `label_assignment(source='cluster', label_space_ver=...)` 中关联记录；
  - 再插入新的 cluster / membership / labels / label_assignment。
- 若算法或配置发生语义性变更（例如阈值大幅变动、构图方式改变），推荐新建 `method` 或新的 `cluster_v2`，旧版结果可按需保留用于对比，但 UI 默认只使用配置的当前 `cluster_*` 版本。
- `source='cluster'` 的 `label_assignment.score` 固定为 `1.0`，真实距离信息通过 `cluster_membership.distance` 暴露，用于排序和分析。

### 4.4 存储落点与重算策略

遵循「cache 存放预处理/模型特征的可复用缓存，data 存放 process 阶段的语义输出」：

- `cache/` 目录：存放预处理与模型前向直接产出的特征/中间结果对应的向量、JSON、缩略图等文件（如 `regions`、`region_embedding` 以及 image embedding）。借助主库中的元数据即可定位这些文件，必要时可通过清空 cache 目录全量重建。
- PostgreSQL 主库（历史文档中曾称 `data/index.db`，现仅作示例路径）：承载 `labels`、`label_aliases`、`label_assignment`（含自动/人工）、聚类结果（`image_similarity_cluster`、`cluster_membership`）、其它 process 计算产物。需要时通过重跑 process（读取 cache 特征）即可再生成。
- 任务层不做跨库 JOIN；process 任务读取 cache、计算后直接写 data。cache 可随时重建，重建后跑一次 process 可恢复 data（人工标签除外）。
- 数据库初始化通过 `scripts/` 目录下的独立脚本一次性创建所需表（原型阶段不做向后兼容），并在 `init_project.sh` 中调用，后续可演进到版本化迁移工具。

### 4.5 PostgreSQL 索引与查询模式（推荐）

为保证在 PostgreSQL 上的查询性能，推荐在初始化脚本中创建以下索引（使用 `IF NOT EXISTS` 以便安全重跑）：

```sql
-- label_assignment：按目标查标签（单图或单 region 的所有标签）
CREATE INDEX IF NOT EXISTS idx_label_assignment_target
  ON label_assignment (target_type, target_id, label_space_ver);

-- label_assignment：按标签聚合（某标签下有哪些图，用于 UI facets）
CREATE INDEX IF NOT EXISTS idx_label_assignment_label
  ON label_assignment (label_id, label_space_ver, source);

-- labels：按 level/key 查找（构建标签树与按 key 查询）
CREATE INDEX IF NOT EXISTS idx_labels_level_key
  ON labels (level, key);

-- image_embedding：按模型过滤（来自现有 M1 表）
CREATE INDEX IF NOT EXISTS idx_image_embedding_model
  ON image_embedding (model_name);

-- region_embedding：按模型过滤
CREATE INDEX IF NOT EXISTS idx_region_embedding_model
  ON region_embedding (model_name);

-- regions：按 image 查 region
CREATE INDEX IF NOT EXISTS idx_regions_image
  ON regions (image_id);

-- cluster_membership：按 target 查其所在 cluster
CREATE INDEX IF NOT EXISTS idx_cluster_membership_target
  ON cluster_membership (target_type, target_id);
```

上述索引覆盖了典型访问模式：  
- 查询某张图片/region 的所有标签；  
- 查询某个标签下有哪些图片；  
- 按模型名称遍历 embedding；  
- 按图片/region 查找其 cluster 归属。

---

## 5. Perception Layer Improvements

M2 在感知层不过度追求全新模型，而是围绕现有 SigLIP / BLIP / OWL‑ViT 做针对性的增强和角色重构。模型与权重版本沿用当前配置（settings 中的默认值），无需更换 checkpoint 名称。

### 5.1 Scene & Boolean Attributes

目标：减少 prompt‑pair 二分类的漂移，更稳定地区分 screenshot / document / has_person / has_text，并让 scene 分类更接近「线性头 + 规则」的模式。

#### 5.1.1 `is_screenshot`

特征组合示意：

- 几何 + 元数据：
  - 宽高比接近常见分辨率（16:9, 19.5:9 等）。
  - EXIF 相机型号缺失或文件名中包含 `Screenshot`/`屏幕快照` 等模式。
- 图像统计：
  - 平均亮度偏高、色块边界清晰（UI 风格）。
- 可选 SigLIP 特征：
  - 与 `"a screenshot of a computer or phone user interface"` prompt 的相似度。

评分方式：

- `score = w1 * aspect_match + w2 * filename_match + w3 * exif_missing + w4 * siglip_score`；
- `score > threshold` 判定 `attr.is_screenshot = True`。

#### 5.1.2 `is_document` / `has_text`

有 OCR 时：

- 若 OCR 文字面积比例 > 某阈值（例如 > 20%），可直接判定为 document。

无 OCR 时：

- 使用边缘密度、直线比例、对比度 + SigLIP 对 `"a document photo"` prompt 的相似度混合成分数。
- `has_text` 允许较宽阈值（以 recall 为主），`is_document` 更严格（以 precision 为主）。

#### 5.1.3 `has_person` / `scene.people`

- 优先引入轻量级人脸 / 人体 detector（Mediapipe/ONNX 等）：
  - 输出：有无脸（has_face）、脸数（face_count）。
- `has_person` 主要由 detector 控制，SigLIP prompt 作为辅助特征。
- scene.people 的决策由 scene classifier 完成，但可以退回到 `scene.other` 以避免误报。

#### 5.1.4 Scene Classifier

长期目标：

- 使用 SigLIP image embedding + 少量标注数据训练一个 **线性分类头**：
  - 类别：`LANDSCAPE / PEOPLE / FOOD / PRODUCT / DOCUMENT / SCREENSHOT / OTHER`。
  - 输入：L2 normalized SigLIP embedding。
  - 输出：7 维 logits / softmax。

落地策略（M2 阶段）：

- **优先复用现有的 `SceneClassifierWithAttributes`**：
  - 保留原有 zero‑shot prompts 与逻辑，并把输出写入：
    - 旧的 `image_scene` 表（兼容 M1）。
    - 新的 `label_assignment`（`level='scene'` 与 `level='attribute'`）。
- 线性头训练（少量标注集）的工作放在 M2.5 / M3：
  - 利用实际 QA 过程中的纠错数据；
  - 保持 label 层接口不变，只更新 classifier 内部实现和 `classifier_version`。

换言之，M2 实现中 scene label pass 的 classifier 具体就是对当前 `SceneClassifierWithAttributes` 的封装，线性头训练与切换作为后续独立迭代，不影响本版 schema 和接口。

#### 5.1.5 特征存储（可选）

上述 face_count、screenshot_score、document_score 等中间特征可以只在 scene/attribute pass 内存中使用；  
为了便于后续系统化调参与评估，推荐在主数据库中增加轻量的 image‑level feature 表，将这些标量以 JSON 形式存储：

```sql
CREATE TABLE image_features (
  image_id      TEXT PRIMARY KEY,
  features_json TEXT NOT NULL,   -- 例如 {"face_count":1,"screenshot_score":0.9,"document_score":0.1}
  updated_at    REAL NOT NULL
);
```

约定：

- `image_features` 属于可重算缓存，只存放由 preprocess / scene pass 直接产出的标量特征；  
- scene / attribute label pass 可以读取或更新该表，评估脚本也可以直接使用其中特征分析阈值与分布；  
- 当需要重建特征时，可以安全地清空该表，再通过重跑相关 pass 回填。

#### 5.1.6 配置与默认值（写入 `config/settings.yaml`）

- `scene.top1_min_score=0.30`：scene classifier 的最低接受置信度，低于则回退 `scene.other`。
- `attr.has_person.face_score_min=0.35`：人脸/人体 detector 的置信度阈值，满足则置为 True。
- `attr.has_text.score_min=0.35`：文字覆盖率/文本倾向得分达到阈值即视为有文本（偏 recall）。
- `attr.is_document.score_min=0.60`：更严格的文档判定阈值（偏 precision）。
- `attr.is_screenshot.score_min=0.55`：综合规则分数达到该值视为截图。
- 以上配置需同步写入 `config/settings.example.yaml`，并在 `config/settings.yaml` 提供默认值。

### 5.2 Region Detection & Region Embedding

M2 对 detection 本身不做大改，而是改变其「角色」：

- **OWL‑ViT 继续沿用** 当前设置：
  - 在 canonical image（去重后代表图）上运行；
  - 使用已有 NMS 和 score 阈值。
- **改造 `_run_region_detection...`**：
  - detection 阶段只负责：
    - 调用 OWL‑ViT 检测；
    - 写入 `regions`（image_id + bbox + detector + raw_label + raw_score）；
    - 裁剪每个 region patch，用 SigLIP 生成 region embedding 并写入 `region_embedding`；
  - 不再在 detection 中确定「最终物体名称」或 refined label（不做 labels re‑ranking）。

这样，所有后续的物体命名 / 聚类 / few‑shot 学习，都只依赖 `regions + region_embedding`，无需重跑 detection。

---

## 6. Label Layer Design

Label 层是 M2 的核心，负责把各种来源的信号统一成「可重算、可追踪、可评估」的标签。

### 6.1 基础标签空间（scene & attributes）

初始化时应在 `labels` 中创建下面的基础条目（示例）：

- Scene labels：
  - `scene.landscape`
  - `scene.snapshot`
  - `scene.people`
  - `scene.food`
  - `scene.product`
  - `scene.document`
  - `scene.screenshot`
  - `scene.other`
- Attribute labels：
  - `attr.has_person`
  - `attr.has_text`
  - `attr.is_document`
  - `attr.is_screenshot`

Scene Label Pass（可映射为单独 Celery 任务，如 `run_label_pass_scene`）：

1. 读取 image embedding + 规则特征 + face detection 结果。
2. 调用 scene classifier（现阶段为 zero‑shot，未来可替换为线性头）得到：
   - scene 类型 + 概率 / 置信度。
   - has_* 布尔属性 + 关联置信度。
3. 写入 `label_assignment`：
   - `source='classifier'`；
   - `label_space_ver='scene_v1'`；
   - `level='scene'` + `level='attribute'`。
4. 保持同步更新 `image_scene`（兼容现有 UI），但未来 UI 逐步改为只依赖 label 层。

**升级路径**：

- 若想在以后新增 scene（例如 `scene.home_office`），只需：
  - 在 `labels` + `label_aliases` 增加相应定义；
  - 更新 scene classifier 训练脚本与权重；
  - 对新的 `label_space_ver` 重算一次 scene pass。

### 6.2 Object Labels（可识别物体）

这一层是 M2 的重点，直接服务「找物」能力（MacBook、iPhone、耳机、披萨等）。

#### 6.2.1 标签空间定义

在 `labels` 中定义 object 标签树，示例：

- `object.electronics`
  - `object.electronics.computer`
  - `object.electronics.laptop`
    - `object.electronics.laptop.macbook`
  - `object.electronics.phone`
  - `object.electronics.earbuds`
    - `object.electronics.earbuds.airpods`
- `object.food`
  - `object.food.pizza`
  - `object.food.noodles`
  - `object.food.burger`

在 `label_aliases` 中为每个 label 定义英文 alias（利于 SigLIP 文本 encoder）：

- `object.food.pizza`：
  - `"pizza"`, `"a pizza"`, `"a slice of pizza on a plate"`
- `object.electronics.laptop.macbook`：
  - `"apple macbook laptop"`, `"a silver macbook pro on a wooden desk"`

#### 6.2.2 文本 embedding 与原型构建

独立 job：`build_object_label_text_prototypes`：

1. 对所有 `level='object'` 的 labels：
   - 聚合其 alias 文本集合；
   - 用 SigLIP 文本 encoder 编成 embedding；
   - 对 alias embedding 做平均，再 L2 归一化，得到 prototype vector；
2. 将所有 prototype 堆成矩阵，并缓存为：
   - `cache/label_text_prototypes/object_v1.npz`
   - 内含：`label_ids`, `label_keys`, `prototypes`。

这一步只在 label 定义更新时需要重跑。

#### 6.2.3 Region‑based Zero‑Shot 物体标签

Object Label Pass（`run_object_label_pass`）流程示意：

1. 读取 `regions + region_embedding`：
   - region 来源：OWL‑ViT 检测；
   - embedding 来源：SigLIP region encoder。
2. 可选 coarse 筛选：
   - 只在 scene ∈ {PRODUCT, FOOD} 的图片上跑 full object label；
   - 人像图片只针对特定标签（比如眼镜、耳机、手机）做子集匹配。
3. 对每个 region embedding 计算与 prototype 的余弦相似度：
   - 只对相关的 label 子集计算（按 scene 或父类别过滤）；
   - 取 top‑k（例如 k=5）；
   - 计算 margin（top1 − top2）。
4. 决策写入：
   - 若 `score >= threshold` 且 `margin >= margin_threshold`：
     - `target_type='region'`, `source='zero_shot'`, `label_space_ver='object_v1'`；
     - 可附带 `extra_json` 记录 rank/margin 等。
   - 同时在 image 级别聚合：
     - 若某 image 的多个 region 重复给出同一 label，可给 image 写 `source='aggregated'` 的 object label，方便按物体过滤整图。

配置与默认值（写入 `config/settings.yaml`，起步值可迭代）：

- `object.zero_shot.top_k=5`
- `object.zero_shot.score_min=0.32`（region embedding 与 prototype 的余弦相似度）
- `object.zero_shot.margin_min=0.08`（top1−top2）
- `object.zero_shot.scene_whitelist=["scene.product","scene.food"]`（其它 scene 可选跑简化标签子集，如手机/耳机）
- `object.aggregation.min_regions=1`、`object.aggregation.score_min=0.32`（满足即可在 image 级写 `source='aggregated'`）
- 以上配置需同步写入 `config/settings.example.yaml`，并在 `config/settings.yaml` 提供默认值。

**当前实现补充：Qwen→SigLIP Object Head（image‑level）**

在 M2 迭代中，我们已经按「teacher‑student 蒸馏」的模式，为 object labels 增加了一颗轻量 SigLIP object head，并接入了 label pass：

- 训练脚本：`tools/train_object_head_from_qwen.py`  
  - 输入：`tmp/ground_truth_auto.jsonl` + `image_embedding` 表中的 SigLIP 全图 embedding；
  - 产出：`models/object_head_from_qwen.pt`，内部包含：
    - `label_keys`: 一组 `object.*` label keys；
    - `state`: `{input_dim,num_labels,state_dict}` 的线性多标签头参数。
- 评估脚本（不依赖 label layer）：
  - `uv run python -m vibe_photos.eval.object_head --gt tmp/ground_truth_human.audited.json`
  - `uv run python -m vibe_photos.eval.object_head --gt tmp/ground_truth_auto.jsonl`
  - 输出 image‑level object head 的 top‑1 / top‑3 / top‑5 命中率，便于快速 sanity‑check 蒸馏质量。
- 集成方式：
  - `vibe_photos.labels.object_label_pass.run_object_label_pass` 仍按本节所述跑 **region‑level zero‑shot + image‑level aggregated**；
  - 若检测到 `models/object_head_from_qwen.pt` 存在，则额外：
    - 读取 SigLIP image embedding（`image_embedding` 表）；
    - 用 object head 预测每个 image 的 object label 概率；
    - 对 `prob >= 0.5` 的 top‑k 结果写入：
      - `target_type='image'`；
      - `label_space_ver=object_v1`；
      - `source='classifier'`；
      - `score=prob`；
    - 匹配不到 DB 中 `labels.key` 的对象会被安全跳过。
  - 为保证重跑幂等，object label pass 在开始时会清理 `object_v1` 下所有：
    - `target_type in ('image','region')` 且
    - `source in ('zero_shot','aggregated','duplicate_propagated','classifier')`
    的记录，再重写当前版本的 labels。

#### 6.2.4 人工 / VLM 辅助（留给 M4）

M2 只在数据结构层面预留：

- 允许用户通过 CLI / 简单 UI 添加 `source='manual'` 的 object labels；
- 在查询层面优先使用 manual 覆盖其它来源；
- 聚类后的 cluster 也可以批量赋予某个 object label（见下文）。

#### 6.2.5 参考实现：原型构建与 Object Label Pass

为了让 M2 中「从 seed → prototype → region zero‑shot → label_assignment」的闭环更易实现，推荐按两个独立 job 来落地 object label 流程，并在 pipeline 中按阶段串联起来。

**一）`build_object_label_text_prototypes`：从 Label + alias → SigLIP 文本原型**

- 建议实现为单独模块（例如 `vibe_photos.labels.object_prototypes`），核心职责：
  1. 扫描 DB 中所有 `level='object'` 且 `is_active=1` 的标签，按 `id` 排序；
  2. 对每个 label：
     - 从 `label_aliases` 中收集 alias 文本，优先使用 `language='en'`，其次 `language='zh'`，若都没有则回退到 `display_name`；
     - 使用 SigLIP 文本 encoder 对 alias 文本编码，得到一批 L2-normalized 文本 embedding；
     - 对同一标签的多条 alias embedding 求平均，再做一次 L2 归一化，得到该标签的 prototype 向量；
  3. 将所有 prototype 堆成矩阵，并缓存到：
     - 路径：`{cache_root}/label_text_prototypes/{output_name}.npz`（例如 `object_v1.npz`）；
     - 内容三列：
       - `label_ids`: `[L] int64`；
       - `label_keys`: `[L] str`；
       - `prototypes`: `[L, D] float32`（所有行 L2-normalized）。
- 该 job 只在 object label 定义（或 alias 集合）更新时需要重跑；实际使用时通过 `load_object_label_text_prototypes` 读取 `.npz` 即可。

**二）`run_object_label_pass`：Region embedding → zero‑shot 打标签**

- 建议实现为独立模块（例如 `vibe_photos.labels.object_label_pass`），核心职责：
  1. 前置条件：
     - detection 阶段已对 canonical image 跑完 OWL‑ViT + SigLIP，并填充了 `regions` 与 `region_embedding` 表；
     - `build_object_label_text_prototypes` 已完成，或在第一次运行时自动触发一次构建。
  2. 启动时：
     - 从 `{cache_root}/label_text_prototypes/{prototype_name}.npz` 载入 `label_ids / label_keys / prototypes`；
     - 从 `labels` 表中一次性拉出这些 `label_ids` 对应的 `Label` 记录，构建 `id → Label` 映射；
     - 根据配置选择要使用的 region embedding 模型名（例如 `settings.models.embedding.resolved_model_name()`）。
  3. 遍历所有待处理的 region embedding：
     - SQL 层面：`RegionEmbedding JOIN Region`，按 `model_name` 过滤，必要时可按 `image_id` 子集过滤；
     - 对于每一条 `(region_id, embedding_path, image_id)`：
       - 从 `{cache_root}/embeddings/{embedding_path}` 载入 `.npy` 向量，确保是一维向量，并做 L2 归一化；
       - 计算与所有 prototypes 的相似度：`sims = prototypes @ region_vec`，得到 `[L]` 的相似度数组；
       - 对 `sims` 做降序排序，取前 `top_k` 作为候选，计算：
         - `top1_score`：最大相似度；
         - `second_score`：次大相似度（若存在），`margin = top1_score − second_score`。
       - 基于 6.2.3 定义的 `score_min` / `margin_min` 做筛选：
         - 若 `top1_score < score_min` 或 `margin < margin_min`，则跳过该 region；
         - 否则可以选择：
           - 仅对 top‑1 标签写入记录；
           - 或对 top‑k 中所有 `score >= score_min` 的标签写入记录。
  4. 写入标签时：
     - Region 级标签：
       - `target_type='region'`，`target_id=region_id`；
       - `source='zero_shot'`，`label_space_ver='object_v1'`；
       - `score=sim`（region embedding 与该 label prototype 的余弦相似度）；
       - `extra_json` 建议包含：
         - `{"sim_rank": idx, "top1_score": top1_score, "margin": margin}`。
     - Image 级聚合标签：
       - `target_type='image'`，`target_id=image_id`；
       - `source='aggregated'`，`label_space_ver='object_v1'`；
       - `score` 可直接复用 region 级的 `sim`，或按多 region 做简单聚合（例如最大值）；
       - `extra_json` 建议包含：
         - `{"from_region": region_id, "sim_rank": idx, "margin": margin}`。
- 典型函数签名可类似：
  - `run_object_label_pass(settings, cache_root, cache_session, label_space_ver="object_v1", prototype_name="object_v1", embedding_model_name=...)`；
  - 其中 `cache_session`（现在与 primary session 相同）负责操作所有相关表（`labels / label_aliases / label_assignment / regions / region_embedding`），缓存根目录仅提供向量/JSON 文件。
- 在 pipeline 串联上，推荐在 detection 阶段（写完 `regions + region_embedding`）之后、scene label pass 之前增加一个 Object Label Pass stage（例如 `run_object_label_pass`），确保下游 clustering 和 QA UI 可以直接使用 object 标签。

### 6.3 标签层约束与合法组合

#### 6.3.1 `target_type` / `target_id` 约定

- `label_assignment.target_type` 只允许两个枚举值：
  - `'image'`：`target_id` 必须等于 `images.image_id`（内容哈希，不因 canonical 或传播而改变）；
  - `'region'`：`target_id` 必须等于 `regions.id`（统一格式：`"{image_id}#{index}"`，见 4.2）。
- 任何其它目标类型（例如文件夹、相册、cluster 本身）不在 M2 范围内，如后续需要会在新版本中扩展。

#### 6.3.2 `labels.level` 与 `target_type` 的合法组合

为避免出现「scene 贴在 region 上」这类语义模糊情况，约定：

- `level='scene'`
  - 只用于 `target_type='image'` 的记录；
  - scene label 不直接贴在 region 上。
- `level='attribute'`
  - 默认只用于 `target_type='image'`（如 `attr.has_person` / `attr.is_document`）；
  - 若未来需要 region‑level 属性（如 `attr.is_blurry_region`），将通过新增 label key 并在 Blueprint 中单独说明的方式引入。
- `level='object'`
  - 既可以贴在 `image` 级（聚合视图），也可以贴在 `region` 级（具体物体框）；
  - 推荐约定：
    - region 级多来自 `source in ('zero_shot','classifier')`；
    - image 级多来自 `source in ('aggregated','manual','duplicate_propagated')`。
- `level='cluster'`
  - 可贴在 `image` 级或 `region` 级，分别对应 image‑level / region‑level cluster；
  - 聚类算法的 `method` 与 `label_space_ver` 决定「这是哪一类簇」。

> M2 不在 DB schema 层面强制上述组合约束，但所有 label pass 的实现必须遵守这些约定，避免写出语义含糊的标签记录。

#### 6.3.3 `labels.key` 与 `parent_id` 语义

- `labels.key` 一旦确定，不重用：
  - 若某标签语义发生重大改变（如 `object.phone.iphone` 从「iPhone」扩展为「所有智能手机」），必须新建 key，而不是就地复用；
  - 旧 key 的历史记录可以保留，用于回溯与评估。
- `parent_id` 只表达标签树结构，不自动做「父标签继承」：
  - 是否给父标签打 assignment 由各 pass 决定；
  - 例如：object label pass 可以选择在子标签命中时，同时为父标签写入 `source='aggregated'` 的标签，也可以完全依赖查询时的「沿树向上聚合」。

---

## 7. Clustering: Similar Object Groups for Unknown Items

M2 的另一大亮点是：即便模型叫不出新物体的名字，也可以依赖 embedding 把「长得像的一堆图」聚在一起。

### 7.1 Goals

- 对于 scene=PRODUCT / FOOD 的图片和 region：
  - 自动发现「相似物体簇」，如同一把键盘、同一台机箱、同一支麦克风；
  - 即便没有任何先验 label 词表，至少能说：
    - 「这是一组看起来非常像的东西」；
  - UI 中可以展示缩略图墙，用户可以给 cluster 命名，例如：
    - “Keychron Kxx 键盘”
    - “那台新的录音机”

### 7.2 Image‑Level Clustering

适用场景：

- 主体占据画面大部分的物体（电脑主机壳、键盘、显示器等）。

输入选择：

- 所有 scene ∈ {PRODUCT, FOOD} 且没有高置信度 object label 的图片；
- 使用去重后的 canonical image（结合 pHash 结果）。

#### 7.2.1 Canonical + pHash：近重复去重与聚类关系

M1 已经通过 pHash（感知哈希）对图片做了“近重复”分组，M2 在此基础上引入「canonical image」的概念，用来减少重复计算、让聚类结果更干净。

约定：

1. **pHash 分组**

   - 所有内容相似度较高的图片（例如连拍、轻微裁剪/调色版本）被分到同一组；
   - 表结构（示例曾用 `data/index.db` 展示；实际部署为 PostgreSQL）记录 exact / near duplicate 关系，示意：
     ```sql
     CREATE TABLE image_near_duplicate_group (
       id                  INTEGER PRIMARY KEY AUTOINCREMENT,
       canonical_image_id  TEXT NOT NULL,
       method              TEXT NOT NULL,   -- 'phash_v1' 等
       created_at          REAL NOT NULL
     );
 
     CREATE TABLE image_near_duplicate_membership (
       group_id            INTEGER NOT NULL,
       image_id            TEXT NOT NULL,
       is_canonical        INTEGER NOT NULL DEFAULT 0,
       distance            REAL NOT NULL,   -- pHash 距离或相似度
       PRIMARY KEY(group_id, image_id),
       FOREIGN KEY(group_id) REFERENCES image_near_duplicate_group(id)
     );
     ```

2. **Canonical image 选择**

   对于每个 pHash 分组，选定 1 张作为 canonical image（原则可以是：

   - 分辨率最高 / 文件体积最大；
   - 若并列则取 `image_id` 字典序最小；
   - 规则可配置，写在 Blueprint 中即可）。

3. **重模型阶段只跑 canonical**

   - Heavy 模型（SigLIP embedding、OWL-ViT detection、region embedding、聚类）**只对 canonical image 执行**；
   - 同组其他图片的 heavy 特征可以在需要时延迟计算，或者通过共享标签的方式间接获得收益。

4. **标签与簇的传播**

   - scene / object / cluster 等标签，在写入 canonical image 之后，可以选择：
     - 只标记 canonical（保留“最干净的真相”）；
     - 或同步到 near duplicate 成员（例如 `source='duplicate_propagated'`），便于检索。
   - 聚类（cluster）阶段推荐：
     - 只对 canonical image 做聚类，得到一组 cluster；
     - 将 cluster label 同步到该 pHash 组内所有图片上，保证“同一物体的所有版本”都被纳入簇。
   - region 级标签默认不传播，可通过配置决定是否覆盖近重复。
   - `images.image_id` 始终为内容哈希，不因 canonical 标记或传播而改变；`regions.id` 统一使用 `{image_id}#{index}`。

这样设计的好处：

- 大幅减少重复计算：3 万张里如果有很多连拍，只需对 canonical 跑重模型；
- 聚类结果更可读：不会因为同一物体的十几张连拍把某个簇“撑爆”；
- 标签传播逻辑简单透明，可在后续版本中按需调整（例如只传播 object 标签，不传播 scene）。

算法建议（简单版本，便于实现和解释）：

1. 取对应图像的 SigLIP embedding（L2 归一化）。
2. 构建 kNN 图：
   - 使用 `NearestNeighbors` / faiss，度量为 cosine 距离；
   - 每个样本取 k 近邻（如 k=20）。
3. 保留相似度 ≥ τ 的边（τ 例如 0.75–0.8）：
   - 相似度 = 1 − cosine_distance。
4. 在该稀疏图上寻找连通分量：
   - 分量大小 ≥ 3 视为有效 cluster。
5. 对每个 cluster：
   - 计算 cluster center（平均 embedding）；
   - 写入 `image_similarity_cluster` + `cluster_membership`（`id` 自增，`key` 同 `labels.key` 采用 `cluster.{method}.{id}`，如 `cluster.siglip_image_knn_v1.17`）；
   - 在 `labels` 中插入 `level='cluster'` 的记录（沿用相同 `key`）；
   - 对每个成员 image 写入 `label_assignment`：
     - `target_type='image'`, `source='cluster'`, `label_space_ver='cluster_v1'`, `score=1.0`（真实距离保留在 `cluster_membership.distance`，排序时再转换）。

配置与默认值（写入 `config/settings.yaml`，可按评估调优）：

- `cluster.image.k=20`、`cluster.image.sim_threshold=0.78`、`cluster.image.min_size=3`
- `cluster.region.k=20`、`cluster.region.sim_threshold=0.82`、`cluster.region.min_size=3`
- 以上配置需同步写入 `config/settings.example.yaml`，并在 `config/settings.yaml` 提供默认值。

### 7.3 Region‑Level Clustering

适用场景：

- 一张图中有多样物体（桌面摆设、机箱内部、货架等）。

输入选择：

- `regions` 中满足：
  - raw_score > 阈值（如 0.4）；
  - 面积占图像的 5%–60%（过滤极小/极大框）。

算法与 image 级基本一致：

1. 取 `region_embedding`（L2 归一化）。
2. 构建 kNN 图并保留相似度 ≥ τ 的边（region embedding 更「聚焦」，阈值可以更高，如 0.8）。
3. 找连通分量，size ≥ 3 为 cluster。
4. 对每个 cluster：
   - 写入 `image_similarity_cluster` 与 `cluster_membership`（`target_type='region'`，`key = cluster.{method}.{id}`，如 `cluster.siglip_region_knn_v1.8`）；
   - 在 `labels` 中插入 `level='cluster'`（沿用相同 `key`），`label_assignment.score` 固定 1.0，距离存于 `cluster_membership.distance`；
   - 给所有 region 成员写 `label_assignment(source='cluster')`；
   - 同时给 region 所在的 image 写 `label_assignment(target_type='image', source='cluster')`，方便 UI。

### 7.4 UI & Naming (Preview)

M2 不要求完整交互 UI，但结构上预留：

- 按 cluster 浏览的视图：
  - 每簇展示若干代表图（中心 + 随机采样）；
  - 展示 cluster size 与部分成员缩略图。
- 用户可以：
  - 给 cluster 改名（直接更新 `labels.display_name`）；
  - 可选：把 cluster 绑定到某个 object label，并将成员全部打上该 object 标签（批量写入 `label_assignment(source='manual')` 或 `source='cluster_promoted'`）。

这部分实际 UI 体验更偏向 M4，但只要 M2 把数据结构和聚类 pass 建好，就已经可以通过 CLI / notebook 做高价值的「找新物」探索。

---

## 8. Evaluation & Calibration Tools

为了判断 M2 是否真的提升了感知与标签质量，需要一套轻量可复用的评估工具。

### 8.1 Labeled Subset

构建一个约 800–1500 张图片的小标注集，格式建议 JSON/JSONL。  
评估标注文件中的每条记录必须包含 `image_id` 字段，与主库中 `images.image_id` 一一对应，可选再附带原始文件名或相对路径作为辅助信息。

示意格式（JSONC）：

```jsonc
{
  "a3f5c8...": {
    "scene": ["scene.product"],
    "attributes": {
      "attr.has_person": false,
      "attr.is_document": false,
      "attr.is_screenshot": false
    },
    "objects": ["object.electronics.computer_case"]
  }  // 其中 "a3f5c8..." 即为 images.image_id
}

也可以使用 JSONL 形式，每行一条记录，例如：

```json
{"image_id": "a3f5c8...", "scene": ["scene.product"], "attributes": {"attr.has_person": false}, "objects": ["object.electronics.computer_case"]}
{"image_id": "b7129d...", "scene": ["scene.food"],    "attributes": {"attr.is_screenshot": false}, "objects": ["object.food.pizza"]}
```

目标：

- 覆盖常见场景：
  - 人像 / 风景 / 食物 / 产品 / 文档 / 截图；
  - 常见电子设备 / 食物 / 日常用品的 object labels；
  - 包含一定量的「新产品 / 冷门设备」用于聚类观察。

### 8.2 CLI Evaluation Script

我们已经实现了一组 CLI 来支撑 M2 的评估与调参与蒸馏：

- **主评估 CLI（label layer）**  
  - `uv run python -m vibe_photos.eval.labels --gt data/labels/ground_truth.json`  
  - 功能：
    - 从 ground truth 文件加载标注；
    - 从数据库中读取 `label_assignment`；
    - 计算：
      - Scene labels：准确率 / confusion matrix；
      - Attributes：precision / recall / F1；
      - Object labels：top‑1 / top‑3 命中率；
      - 聚类：对于标注过的物体，观察其所属 cluster 的 purity（可选）。

- **属性 head 阈值扫面（attribute threshold tuning）**  
  - `uv run python -m vibe_photos.eval.attribute_thresholds --gt tmp/ground_truth_human.audited.json`  
  - 直接读取：
    - Qwen 蒸馏出的 attribute head（`models/attribute_head_from_qwen.pt`）；
    - 人工 GT 中的 `attr.has_person` / `attr.has_text` / `attr.has_animal`；
  - 对每个 attribute 扫一系列 logits→prob 阈值，输出：
    - 不同阈值下的 precision / recall / F1；
    - 推荐的「高 precision」点，用于更新 `config/settings.yaml` 中的 `attributes.head_thresholds`。

- **Object head 质量评估（student vs teacher / human）**  
  - `uv run python -m vibe_photos.eval.object_head --gt tmp/ground_truth_auto.jsonl`  
  - `uv run python -m vibe_photos.eval.object_head --gt tmp/ground_truth_human.audited.json`  
  - 直接用 object head（`models/object_head_from_qwen.pt`）对 SigLIP image embedding 做预测，计算：
    - image‑level object top‑1 / top‑3 / top‑5 命中率；
  - 不经过 label layer，仅用于校验 Qwen→SigLIP 蒸馏质量。

- **Object 错误分析（label layer vs GT）**  
  - `uv run python -m vibe_photos.eval.object_errors --gt tmp/ground_truth_human.audited.json --output-dir tmp`  
  - 读取 label layer 中的 object labels（包括 zero‑shot 聚合 + learned head + 手工标签），与 GT 对比：
    - 统计最终 top‑1 / top‑3 / top‑5 命中率；
    - 将 top‑3 未命中的样本写入 `eval_object_errors.jsonl`，包含：
      - `image_id`；
      - `gt_objects`；
      - `pred_topk`（pipeline 输出的前若干 object labels）；
      - `hit_top1` / `hit_top3` / `hit_top5`。
  - 该 JSONL 可作为 notebook / GPT 分析的输入，用于发现系统性错误模式。

默认评估只针对 canonical image；如需评估 near‑duplicate 标签传播质量，可以额外定义「duplicate consistency」指标，对每个 near‑duplicate 分组比较 canonical 与成员图片在 scene / object / attribute 上的一致性。

默认评估只针对 canonical image；  
如需评估 near‑duplicate 标签传播质量，可以额外定义「duplicate consistency」指标，对每个 near‑duplicate 分组比较 canonical 与成员图片在 scene / object / attribute 上的一致性。

推荐查询模式（示意 SQL）：

- Scene（单图的 scene 预测）：

  ```sql
  SELECT l.key, la.score
  FROM label_assignment la
  JOIN labels l ON la.label_id = l.id
  WHERE la.target_type = 'image'
    AND la.target_id = :image_id
    AND la.label_space_ver = :scene_current
    AND l.level = 'scene';
  ```

- Attributes（单图的布尔属性）：

  ```sql
  SELECT l.key, la.score
  FROM label_assignment la
  JOIN labels l ON la.label_id = l.id
  WHERE la.target_type = 'image'
    AND la.target_id = :image_id
    AND la.label_space_ver = :scene_current
    AND l.level = 'attribute';
  ```

- Object（按图评估 object labels，默认聚合 image‑level）：

  ```sql
  SELECT l.key, la.score
  FROM label_assignment la
  JOIN labels l ON la.label_id = l.id
  WHERE la.target_type = 'image'
    AND la.target_id = :image_id
    AND la.label_space_ver = :object_current
    AND l.level = 'object'
    AND la.source IN ('aggregated','manual','duplicate_propagated');
  ```

- Cluster（可选，用于检查某图的 cluster 归属）：

  ```sql
  SELECT l.key
  FROM label_assignment la
  JOIN labels l ON la.label_id = l.id
  WHERE la.target_type = 'image'
    AND la.target_id = :image_id
    AND la.label_space_ver = :cluster_current
    AND l.level = 'cluster';
  ```

### 8.3 Analysis Helpers

附加工具（CLI 或 notebook）：

- 给定某个 label（scene/object/attribute），列出：
  - **最常见的误判样本**；
  - **得分接近阈值的边缘样本**。
- 用于调节：
  - 阈值（score/margin）；
  - label alias 文案；
  - scene/object label 空间本身。

---

## 9. Recommended Implementation Plan (M2 Roadmap)

本节将 M2 拆成可以逐步落地的几个阶段，尽量保持每一步「增量」「可回滚」。

主流程建议：process 任务直接读取 cache 目录中的特征/中间结果（通过数据库元数据定位），计算后写入主库（不跨库 JOIN）；若重建 cache，仅需重跑 preprocess + process，即可恢复 data（人工标签除外）。

### 9.1 M2‑A: 统一标签层（Label Layer Foundation）

**目标**：把所有 scene / has_person / object / cluster 标签统一收敛到 `labels + label_aliases + label_assignment` 三张表。

- **工作项**：
  - [ ] 在 DB 中增加：
    - `labels`
    - `label_aliases`
    - `label_assignment`
  - [ ] 实现 `LabelRepository`（统一操作三表的封装层）：
    - `get_or_create_label(key, level, display_name, parent_key)`；
    - `ensure_aliases(label, aliases, language)`；
    - `upsert_label_assignment(...)`。
  - [ ] 初始化基础 label 集：
    - Scene：`scene.landscape / scene.snapshot / scene.people / scene.food / scene.product / scene.document / scene.screenshot / scene.other`；
    - Attributes：`attr.has_person / attr.has_text / attr.is_document / attr.is_screenshot`；
    - Object：选一小批高优先级物体（MacBook / iPhone / AirPods / pizza / burger 等）起步。

**完成度判断**：

- 能用一两条 SQL 查询出任一 image 的 scene / attributes / object / cluster 标签；
- 现有 scene 字段（`image_scene`）仍正常更新，但已经不再是「唯一来源」。

### 9.2 M2‑B: Detection Pass → Region + Region Embedding Only

**目标**：让 detection 阶段退居「特征工厂」，稳定输出 region + region embedding，而不直接承担标签决策。

- **工作项**：
  - [ ] 重构 `_run_region_detection_and_reranking`：
    - 只做：OWL‑ViT 检测 + NMS + priority + 写 `regions` 表；
    - 裁剪 patch + SigLIP → 写 `region_embedding` 表；
    - 不再计算 refined_label / refined_score。
  - [ ] 保留 JSON cache（便于 debug），但标签信息逐步移交给 label 层。

**完成度判断**：

- 在不跑任何 label pass 的情况下，`regions + region_embedding` 已能完整反映 detection + region 特征；
- 旧 UI 至少能继续显示 bbox 信息（可以临时仍用旧的字段 / JSON）。

为了平滑地从 M1 迁移到 M2，避免长时间功能倒退，推荐遵循下面的 detection / canonical 迁移顺序：

1. **先稳定 canonical 机制（不改 detection 输出）**：  
   - 引入 pHash 分组与 canonical 标记（见 7.2.1），仍然使用现有 detection / scene 结果。
2. **在 canonical image 上增加新的 detection pipeline**：  
   - 仅对 canonical image 跑 OWL‑ViT + SigLIP，写入 `regions` 与 `region_embedding`；  
   - 同期继续保留旧的 detection 结构（例如 JSON cache 或旧表），作为过渡期 fallback。
3. **将 object label pass / clustering 完全迁移到新特征上**：  
   - 新的 object label pass 与 cluster pass 只依赖 `regions + region_embedding` 与 image/region embedding；  
   - Web UI / QA 工具从 label 层读取 object / cluster 标签，而不再直接依赖旧 detection 的 refined label。
4. **最后按需清理 legacy detection 产物**：  
   - 在确认新 pipeline 稳定后，可以停止写入旧 detection 字段/表，并视需要通过一次性脚本清理历史遗留数据。

### 9.3 M2‑C: Object Label Pass（核心收益点）

**目标**：基于 region embedding + 文本 prototype zero‑shot 输出 object labels，并在 image 层做聚合。

- **工作项**：
  - [ ] 定义并初始化 object label 列表与 alias（针对最关心的设备和食物）。
  - [ ] 实现 `build_object_label_text_prototypes`：
    - 从 DB 读取 `level='object'` 的 labels + alias；
    - 用 SigLIP 文本 encoder 生成 prototype 并缓存。
  - [ ] 实现 `run_object_label_pass`：
    - 遍历 `regions + region_embedding`；
    - 计算相似度，基于 score/margin/scene 过滤；
    - 为 region 写 `source='zero_shot'` 的 object label；
    - 为 image 写 `source='aggregated'` 的 object label。
  - [ ] 在 QA UI 或简单 CLI 中增加 object 标签视图：
    - 单图页面显示 object 标签列表；
    - 简单过滤：按 `object.*` 标签筛图。

**完成度判断**：

- 简单查询即可以「MacBook / iPhone / pizza / burger / keyboard / earphones」为关键词筛出合理图片集合；
- 少量人工 spot‑check 评估后，阈值和 alias 文案基本稳定。

### 9.4 M2‑D: Scene Label Pass 重构（先复用现有 classifier）

**目标**：将现有 scene classifier 输出映射为 label 层记录，避免后续迁移时割裂。

- **工作项**：
  - [ ] 封装 `run_scene_label_pass`：
    - 调用现有 `SceneClassifierWithAttributes`；
    - 同步更新：
      - `image_scene` 表（兼容性）；
      - `label_assignment` 中 `level='scene'` 与 `level='attribute'` 标签（`source='classifier'`）。
  - [ ] Web UI 改造：
    - scene filters 基于 `label_assignment` 聚合，而不是直接读 `image_scene`。

**完成度判断**：

- M1 的 scene / 布尔属性查询在 M2 代码路径上仍然可用；
- 新增的 label 层能够完整复现现有筛选功能。

### 9.5 M2‑E: Clustering Pass（相似物体簇）

**目标**：对 image / region embedding 做聚类，形成便于「发现新物体」的簇，并记入标签系统。

- **工作项**：
  - [ ] 实现 image‑level clustering：
    - 仅针对 scene ∈ {PRODUCT, FOOD} 且缺乏强 object label 的图片；
    - 使用 SigLIP image embedding + kNN + 连通分量聚类；
    - 写入 `image_similarity_cluster` + `cluster_membership`；
    - 同步为 `level='cluster'` 标签 + `label_assignment(source='cluster')`。
  - [ ] 实现 region‑level clustering（可选，优先级略低）：
    - 使用 `region_embedding` 做类似聚类；
    - `target_type='region'` + 对应 image 聚合标签。
  - [ ] 在 QA 工具或 UI 中增加「按 cluster 浏览」入口：
    - 每簇显示若干代表图与 size。

**完成度判断**：

- 对于未被 object label 覆盖的新设备 / 冷门产品，能在 cluster 视图中形成明显的一两个簇；
- 用户主观感觉「这一簇就是同一个东西的不同照片」。

---

## 10. Summary

M2 的设计核心是三点：

- **特征与标签解耦**：SigLIP / BLIP / OWL‑ViT 等模型只负责稳定输出 embedding 与 bbox，所有语义标签均通过统一 label 层生成。
- **标签层统一与版本化**：使用 `labels + label_aliases + label_assignment` 管理 scene / object / attribute / cluster 标签，记录 source 与 label_space_ver，支持重算与多来源融合。
- **面向「找物」与「未知新物体」**：优先实现 region‑based object labels 与 embedding clustering，使得常见物体识别与新产品聚合都达到「日常可用」的水准。

在此基础上，后续 M3/M4 可以：

- 迁移到 PostgreSQL + pgvector（搜索与 API）；
- 引入线性 scene 头与小分类器；
- 加入 UI 级的标注与 few‑shot 学习，而无需再重构底层感知与标签系统。
