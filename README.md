# AI Photo Classifier

使用 [OpenCLIP](https://github.com/mlfoundations/open_clip) 模型對旅行照片進行多標籤分類的工具。可自動識別照片中的場景特徵（如山、海、寺廟）與氛圍（如浪漫、靜謐、神秘），並支援從 EXIF 讀取 GPS 定位資訊進行地點識別。

## 功能特色

- **多標籤分類**：一張照片可同時識別多個特徵
- **自訂標籤**：可透過 JSON 檔案自訂分類標籤
- **GPS 地點識別**：自動讀取照片 EXIF 中的 GPS 座標並反向地理編碼
- **批量處理**：支援整個資料夾的照片批量分類
- **自動整理**：可依分類結果自動將照片整理到對應資料夾
- **多種輸出格式**：支援 JSON 和 CSV 格式

## 系統需求

- Python 3.8+
- 建議使用 Apple Silicon Mac (M1/M2/M3) 或 NVIDIA GPU 以獲得更好的效能
- 記憶體：至少 4GB（使用 ViT-L-14 模型建議 8GB 以上）

## 支援的圖片格式

| 類別 | 副檔名 | 說明 |
|------|--------|------|
| **常見格式** | `.jpg` `.jpeg` `.png` `.webp` `.gif` `.bmp` | 直接支援 |
| **現代格式** | `.avif` `.heic` `.heif` | HEIC 需安裝 `pillow-heif` |
| **專業格式** | `.tiff` `.tif` `.jp2` `.jpx` `.dng` | DNG 需安裝 `rawpy` |

**安裝額外格式支援：**
```bash
pip install pillow-heif  # 支援 iPhone HEIC 格式
pip install rawpy        # 支援 RAW/DNG 格式
```

## 安裝

```bash
cd proj-ai-pic
pip install -r requirements.txt
```

## 專案結構

```
proj-ai-pic/
├── src/                        # 源代碼
│   └── classify_photos.py      # 主程式
├── config/                     # 配置文件
│   └── labels_example.json     # 標籤配置範例
├── samples/                    # 範例照片
├── output/                     # 輸出結果
└── requirements.txt            # 依賴套件
```

## 快速開始

```bash
# 基本使用：分類照片並輸出 JSON
python3 src/classify_photos.py samples/ --output output/result.json

# 使用較精準的模型
python3 src/classify_photos.py samples/ \
  --model ViT-L-14 \
  --pretrained laion2b_s32b_b82k \
  --output output/result.json
```

## 參數說明

### 必要參數

| 參數 | 說明 |
|------|------|
| `input_dir` | 照片所在目錄路徑 |

### 模型參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--model` | `ViT-B-32` | OpenCLIP 模型名稱 |
| `--pretrained` | `laion2b_s34b_b79k` | 預訓練權重名稱 |

**常用模型組合：**

| 模型 | 參數量 | 速度 | 精準度 | 適用場景 |
|------|--------|------|--------|----------|
| `ViT-B-32` | 151M | 快 | 一般 | 快速預覽、資源有限 |
| `ViT-L-14` | 428M | 中 | 高 | 平衡速度與精準度（推薦） |
| `ViT-H-14` | 986M | 慢 | 很高 | 追求最佳精準度 |

### 輸出參數

| 參數 | 說明 |
|------|------|
| `--output`, `-o` | 輸出檔案路徑 |
| `--format`, `-f` | 輸出格式：`json`（預設）或 `csv` |

### 標籤參數

| 參數 | 說明 |
|------|------|
| `--labels-file`, `-l` | 自訂標籤配置檔（JSON 格式） |
| `--threshold`, `-t` | 標籤分數閾值，低於此值不顯示（預設：0.05） |
| `--save-labels` | 將預設標籤儲存為 JSON 檔案供修改 |

### 整理參數

| 參數 | 說明 |
|------|------|
| `--organize` | 根據分類結果整理照片到資料夾 |
| `--organize-dir` | 整理後的輸出目錄（預設：`input_dir/organized`） |
| `--move` | 移動而非複製照片（與 `--organize` 搭配） |

### 地點識別參數

| 參數 | 說明 |
|------|------|
| `--extract-location` | 從照片 EXIF 讀取 GPS 並識別地點名稱 |

## 使用範例

### 範例 1：基本分類

```bash
python3 src/classify_photos.py ~/Photos/Japan2024 --output output/japan.json
```

**輸出結果：**
```json
{
  "tokyo-tower.jpg": {
    "scores": {
      "建築": 0.65,
      "城市": 0.25,
      "夕陽": 0.08
    },
    "top_label": "建築"
  }
}
```

### 範例 2：使用高精準度模型

```bash
python3 src/classify_photos.py ~/Photos/Travel \
  --model ViT-L-14 \
  --pretrained laion2b_s32b_b82k \
  --output output/travel.json
```

### 範例 3：使用自訂標籤

編輯 `config/labels_example.json`（標籤名稱可用中文）：
```json
{
  "海灘": "a photo of beach, ocean waves, sandy shore, tropical",
  "高山": "a photo of mountains, hiking, alpine scenery",
  "在地美食": "a photo of delicious food, local cuisine, restaurant",
  "璀璨夜景": "night view, city lights, illumination"
}
```

執行：
```bash
python3 src/classify_photos.py ~/Photos \
  --labels-file config/labels_example.json \
  --output output/result.json
```

### 範例 4：自動整理照片到資料夾

```bash
python3 src/classify_photos.py ~/Photos/Unsorted \
  --model ViT-L-14 \
  --pretrained laion2b_s32b_b82k \
  --organize \
  --organize-dir ~/Photos/Organized
```

**執行後資料夾結構（使用中文標籤命名）：**
```
~/Photos/Organized/
├── 山景/
│   ├── IMG_001.jpg
│   └── IMG_002.jpg
├── 海洋/
│   └── IMG_003.jpg
├── 寺廟/
│   └── IMG_004.jpg
└── 美食/
    └── IMG_005.jpg
```

### 範例 5：輸出 CSV 格式（方便 Excel 開啟）

```bash
python3 src/classify_photos.py ~/Photos \
  --output output/result.csv \
  --format csv
```

### 範例 6：提取照片 GPS 地點資訊

```bash
python3 src/classify_photos.py ~/Photos/Japan \
  --extract-location \
  --output output/japan_with_location.json
```

**輸出結果（有 GPS 資訊）：**
```json
{
  "shrine.jpg": {
    "scores": {
      "寺廟": 0.82,
      "靜謐": 0.12
    },
    "top_label": "寺廟",
    "gps": {
      "lat": 35.6762,
      "lon": 139.6503
    },
    "location": {
      "country": "日本",
      "city": "東京都",
      "district": "渋谷区",
      "display_name": "明治神宮, 渋谷区, 東京都, 日本"
    }
  }
}
```

**輸出結果（無 GPS 資訊）：**
```json
{
  "photo.jpg": {
    "scores": {
      "山景": 0.75
    },
    "top_label": "山景",
    "gps": "-",
    "location": "-"
  }
}
```

**GPS 輸出格式說明：**

| GPS 狀態 | `gps` 欄位 | `location` 欄位 |
|----------|------------|-----------------|
| 無 GPS 資訊 | `"-"` | `"-"` |
| 有 GPS，地理編碼成功 | `{"lat": 25.0, "lon": 121.5}` | `{"country": "...", "city": "...", ...}` |
| 有 GPS，地理編碼失敗 | `{"lat": 25.0, "lon": 121.5}` | `"-"` |

> **注意**：許多照片在傳輸過程中（如通訊軟體、雲端壓縮）會遺失 GPS 資訊。建議使用 USB 直接複製或選擇「原始畫質」下載。

### 範例 7：只顯示高信心度的標籤

```bash
python3 src/classify_photos.py ~/Photos \
  --threshold 0.2 \
  --output output/result.json
```

### 範例 8：匯出預設標籤供修改

```bash
python3 src/classify_photos.py --save-labels config/my_labels.json
```

## 預設標籤

共 56 個旅行相關標籤，涵蓋：

| 類別 | 標籤 |
|------|------|
| 自然景觀 | `山景`, `海洋`, `森林`, `湖泊`, `夕陽`, `天空`, `瀑布`, `沙漠`, `雪景`, `極光`, `星空` |
| 氛圍感受 | `浪漫`, `靜謐`, `狂野`, `自由`, `神秘`, `放鬆` |
| 建築景點 | `建築`, `城市`, `寺廟`, `城堡`, `博物館`, `橋樑`, `世界遺產` |
| 人物活動 | `人像`, `團體`, `活動`, `極限運動`, `健行`, `單車`, `露營` |
| 美食體驗 | `美食`, `街頭小吃`, `咖啡廳`, `酒莊` |
| 住宿交通 | `度假村`, `溫泉`, `郵輪`, `火車`, `纜車` |
| 旅遊類型 | `親子`, `蜜月`, `背包客`, `奢華`, `文化體驗` |

## 自訂標籤格式

標籤配置檔為 JSON 格式：

```json
{
  "標籤名稱（中文 OK）": "英文描述（用於 AI 比對）"
}
```

### 標籤名稱 vs 描述

| 欄位 | 用途 | 語言 |
|------|------|------|
| **Key（標籤名稱）** | 輸出結果顯示、資料夾命名 | **中文 OK** ✅ |
| **Value（描述）** | 送給 AI 模型比對 | **建議英文** ⚠️ |

> **為什麼描述要用英文？**
> OpenCLIP 模型主要使用英文資料訓練，英文描述能獲得更準確的辨識結果。
> 標籤名稱不會送給模型，純粹是給人看的，所以可以用任何語言。

### 範例

```json
{
  "山景": "a photo of mountains, mountain landscape, alpine scenery",
  "海灘": "a photo of beach, ocean waves, sandy shore, tropical",
  "夜景": "a photo of night view, city lights, illumination",
  "溫泉": "a photo of hot spring, onsen, spa, thermal bath"
}
```

**輸出結果會顯示中文標籤：**
```json
{
  "photo.jpg": {
    "scores": {
      "寺廟": 0.77,
      "靜謐": 0.13
    },
    "top_label": "寺廟"
  }
}
```

**整理資料夾也會使用中文名稱：**
```
~/Photos/Organized/
├── 山景/
├── 海洋/
├── 寺廟/
└── 美食/
```

### 撰寫技巧

- 描述越詳細，辨識越準確
- 可使用逗號分隔多個同義詞
- 加入相關的英文關鍵字提升辨識率

## 效能建議

| 裝置 | 建議模型 | 預估速度 |
|------|----------|----------|
| Apple M1/M2/M3 | ViT-L-14 | ~5 張/秒 |
| NVIDIA GPU | ViT-H-14 | ~10 張/秒 |
| CPU only | ViT-B-32 | ~1 張/秒 |

## 注意事項

1. **GPS 地點識別**：需要照片保留原始 EXIF 資訊，轉檔或壓縮可能導致 GPS 資料遺失
2. **首次執行**：模型會自動下載（約 1-2GB），請確保網路連線穩定

### 地點識別 API 說明

地點識別使用 [OpenStreetMap Nominatim API](https://nominatim.org/)（免費服務）：

| 項目 | 說明 |
|------|------|
| **速率限制** | 每秒 1 次請求（腳本已自動控制） |
| **重試機制** | 失敗時自動重試最多 3 次 |
| **容錯處理** | API 錯誤不會中斷整體流程，該照片地點欄位顯示 `"-"` |
| **離線運作** | 無網路時分類功能正常，僅地點識別無法使用 |

**執行結束會顯示統計：**
```
=== GPS 統計 ===
  有 GPS 資訊: 50 張
  地點識別成功: 48 張
  地點識別失敗: 2 張（API 錯誤或逾時）
```

