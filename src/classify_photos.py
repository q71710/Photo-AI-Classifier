#!/usr/bin/env python3
"""
旅行照片 AI 分類工具
使用 OpenCLIP 模型對照片進行多標籤分類

使用方式:
    python classify_photos.py /path/to/photos --output results.json
    python classify_photos.py /path/to/photos --organize  # 自動整理到資料夾
"""

import argparse
import json
import csv
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from tqdm import tqdm

import open_clip

try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False


# 預設標籤配置（可透過 --labels-file 自訂）
DEFAULT_LABELS = {
    # 自然景觀
    "mountain": "a photo of mountains, mountain landscape",
    "ocean": "a photo of ocean, sea, beach, coastal scenery",
    "forest": "a photo of forest, trees, woodland",
    "lake": "a photo of lake, river, water reflection",
    "sunset": "a photo of sunset, sunrise, golden hour",
    "sky": "a photo of beautiful sky, clouds, blue sky",

    # 氛圍與感受
    "romantic": "a romantic scene, love, couple, intimate atmosphere",
    "peaceful": "a peaceful and quiet scene, serene, tranquil, calm",
    "wild": "a wild and adventurous scene, extreme, exciting",
    "freedom": "a scene of freedom, open space, liberation, vast",
    "mysterious": "a mysterious atmosphere, foggy, dark, enigmatic",

    # 建築與城市
    "architecture": "a photo of architecture, buildings, historic structure",
    "city": "a photo of city, urban landscape, street scene",
    "temple": "a photo of temple, church, religious building",

    # 人物與活動
    "portrait": "a portrait photo of person, people, face",
    "group": "a group photo, friends, family gathering",
    "food": "a photo of food, cuisine, restaurant, dining",
    "activity": "people doing outdoor activity, sports, adventure",
}


def get_gps_from_exif(image_path: Path) -> Optional[tuple]:
    """從圖片 EXIF 讀取 GPS 座標"""
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None

        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value

        if not gps_info:
            return None

        # 解析經緯度
        def convert_to_degrees(value):
            d, m, s = value
            return float(d) + float(m) / 60 + float(s) / 3600

        lat = convert_to_degrees(gps_info.get("GPSLatitude", (0, 0, 0)))
        if gps_info.get("GPSLatitudeRef", "N") == "S":
            lat = -lat

        lon = convert_to_degrees(gps_info.get("GPSLongitude", (0, 0, 0)))
        if gps_info.get("GPSLongitudeRef", "E") == "W":
            lon = -lon

        if lat == 0 and lon == 0:
            return None

        return (lat, lon)
    except Exception:
        return None


def reverse_geocode(lat: float, lon: float, geolocator, max_retries: int = 3) -> Optional[dict]:
    """反向地理編碼：座標 -> 地點資訊（含重試機制）"""
    for attempt in range(max_retries):
        try:
            location = geolocator.reverse(f"{lat}, {lon}", language="zh-TW", timeout=10)
            if location and location.raw:
                addr = location.raw.get("address", {})
                return {
                    "country": addr.get("country"),
                    "city": addr.get("city") or addr.get("town") or addr.get("county"),
                    "district": addr.get("suburb") or addr.get("district"),
                    "display_name": location.address,
                }
            return None
        except GeocoderTimedOut:
            if attempt < max_retries - 1:
                time.sleep(2)  # 等待後重試
                continue
        except Exception as e:
            # 網路錯誤、API 錯誤等，記錄但不中斷
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
    return None


def load_model(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
    """載入 OpenCLIP 模型"""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    else:
        device = "cpu"
    print(f"使用裝置: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer, device


def load_labels(labels_file: Optional[Path] = None) -> dict:
    """載入標籤配置"""
    if labels_file and labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_LABELS


def encode_labels(labels: dict, model, tokenizer, device) -> tuple:
    """編碼所有標籤為特徵向量"""
    label_names = list(labels.keys())
    label_texts = list(labels.values())

    text_tokens = tokenizer(label_texts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return label_names, text_features


def classify_image(
    image_path: Path,
    model,
    preprocess,
    text_features,
    label_names: list,
    device: str,
    threshold: float = 0.1,
) -> dict:
    """對單張圖片進行分類"""
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": str(e), "scores": {}}

    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 計算相似度分數
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    scores = {}
    for name, score in zip(label_names, similarity[0].cpu().numpy()):
        if score >= threshold:
            scores[name] = float(score)

    # 按分數排序
    scores = dict(sorted(scores.items(), key=lambda x: -x[1]))

    return {"scores": scores, "top_label": next(iter(scores), None)}


def find_images(input_dir: Path) -> list:
    """尋找所有圖片檔案"""
    # 常見圖片格式（PIL 支援更多，這裡列出最常用的）
    extensions = {
        # 常見格式
        ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
        # 現代格式
        ".avif", ".heic", ".heif",
        # 專業格式
        ".tiff", ".tif", ".jp2", ".jpx",
        # RAW 格式（需額外套件支援）
        ".dng",
    }
    images = []

    for ext in extensions:
        images.extend(input_dir.rglob(f"*{ext}"))
        images.extend(input_dir.rglob(f"*{ext.upper()}"))

    return sorted(set(images))


def classify_all(
    input_dir: Path,
    model,
    preprocess,
    tokenizer,
    device: str,
    labels: dict,
    threshold: float = 0.1,
    extract_location: bool = False,
) -> dict:
    """對目錄中所有圖片進行分類"""
    label_names, text_features = encode_labels(labels, model, tokenizer, device)

    images = find_images(input_dir)
    print(f"找到 {len(images)} 張圖片")

    # 初始化地理編碼器
    geolocator = None
    if extract_location and GEOPY_AVAILABLE:
        geolocator = Nominatim(user_agent="photo_classifier")
        print("已啟用 GPS 地點識別")
    elif extract_location and not GEOPY_AVAILABLE:
        print("警告: 請安裝 geopy 以啟用地點識別 (pip install geopy)")

    results = {}
    location_count = 0
    gps_count = 0
    geocode_fail_count = 0

    for image_path in tqdm(images, desc="分類中"):
        rel_path = str(image_path.relative_to(input_dir))
        result = classify_image(
            image_path, model, preprocess, text_features, label_names, device, threshold
        )

        # 讀取 GPS 並反向地理編碼
        if extract_location:
            gps = get_gps_from_exif(image_path)
            if gps:
                gps_count += 1
                result["gps"] = {"lat": gps[0], "lon": gps[1]}
                if geolocator:
                    try:
                        location = reverse_geocode(gps[0], gps[1], geolocator)
                        if location:
                            result["location"] = location
                            location_count += 1
                        else:
                            result["location"] = "-"
                            geocode_fail_count += 1
                    except Exception:
                        # 確保任何未預期的錯誤都不會中斷流程
                        result["location"] = "-"
                        geocode_fail_count += 1
                    time.sleep(1)  # Nominatim API 限制: 1 請求/秒
            else:
                result["gps"] = "-"
                result["location"] = "-"

        results[rel_path] = result

    if extract_location:
        print(f"\n=== GPS 統計 ===")
        print(f"  有 GPS 資訊: {gps_count} 張")
        print(f"  地點識別成功: {location_count} 張")
        if geocode_fail_count > 0:
            print(f"  地點識別失敗: {geocode_fail_count} 張（API 錯誤或逾時）")

    return results


def save_json(results: dict, output_path: Path):
    """儲存為 JSON 格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"結果已儲存至: {output_path}")


def save_csv(results: dict, output_path: Path, labels: dict):
    """儲存為 CSV 格式"""
    label_names = list(labels.keys())

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "top_label"] + label_names)

        for filename, data in results.items():
            scores = data.get("scores", {})
            row = [filename, data.get("top_label", "")]
            row.extend([scores.get(label, 0) for label in label_names])
            writer.writerow(row)

    print(f"結果已儲存至: {output_path}")


def organize_photos(results: dict, input_dir: Path, output_dir: Path, copy: bool = True):
    """根據分類結果整理照片到資料夾"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, data in tqdm(results.items(), desc="整理照片"):
        top_label = data.get("top_label")
        if not top_label:
            top_label = "unclassified"

        src = input_dir / filename
        dest_dir = output_dir / top_label
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / Path(filename).name

        if copy:
            shutil.copy2(src, dest)
        else:
            shutil.move(src, dest)

    print(f"照片已整理至: {output_dir}")


def print_summary(results: dict):
    """印出分類摘要"""
    label_counts = {}
    location_counts = {}

    for data in results.values():
        top_label = data.get("top_label")
        if top_label:
            label_counts[top_label] = label_counts.get(top_label, 0) + 1

        location = data.get("location")
        if isinstance(location, dict):
            country = location.get("country")
            city = location.get("city")
        else:
            country = None
            city = None
        if country and city:
            loc_key = f"{country} - {city}"
            location_counts[loc_key] = location_counts.get(loc_key, 0) + 1

    print("\n=== 分類摘要 ===")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count} 張")
    print(f"  總計: {len(results)} 張")

    if location_counts:
        print("\n=== 地點摘要 ===")
        for loc, count in sorted(location_counts.items(), key=lambda x: -x[1]):
            print(f"  {loc}: {count} 張")


def main():
    parser = argparse.ArgumentParser(
        description="使用 AI 分類旅行照片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 基本分類，輸出 JSON
  python classify_photos.py ./photos --output results.json

  # 分類並自動整理到資料夾
  python classify_photos.py ./photos --organize --organize-dir ./organized

  # 使用自訂標籤
  python classify_photos.py ./photos --labels-file my_labels.json

  # 輸出 CSV 格式
  python classify_photos.py ./photos --output results.csv --format csv
        """,
    )

    parser.add_argument("input_dir", type=Path, help="照片所在目錄")
    parser.add_argument("--output", "-o", type=Path, help="輸出檔案路徑")
    parser.add_argument(
        "--format", "-f",
        choices=["json", "csv"],
        default="json",
        help="輸出格式 (預設: json)"
    )
    parser.add_argument(
        "--labels-file", "-l",
        type=Path,
        help="自訂標籤配置檔 (JSON 格式)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.05,
        help="標籤分數閾值，低於此值不顯示 (預設: 0.05)"
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="根據分類結果整理照片到資料夾"
    )
    parser.add_argument(
        "--organize-dir",
        type=Path,
        help="整理後的輸出目錄 (預設: input_dir/organized)"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="移動而非複製照片 (與 --organize 搭配使用)"
    )
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        help="OpenCLIP 模型名稱 (預設: ViT-B-32)"
    )
    parser.add_argument(
        "--pretrained",
        default="laion2b_s34b_b79k",
        help="預訓練權重 (預設: laion2b_s34b_b79k)"
    )
    parser.add_argument(
        "--save-labels",
        type=Path,
        help="將預設標籤儲存為 JSON 檔案供自訂修改"
    )
    parser.add_argument(
        "--extract-location",
        action="store_true",
        help="從照片 EXIF 讀取 GPS 並識別地點名稱"
    )

    args = parser.parse_args()

    # 儲存預設標籤
    if args.save_labels:
        with open(args.save_labels, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_LABELS, f, ensure_ascii=False, indent=2)
        print(f"預設標籤已儲存至: {args.save_labels}")
        return

    # 檢查輸入目錄
    if not args.input_dir.exists():
        print(f"錯誤: 找不到目錄 {args.input_dir}")
        return

    # 載入模型與標籤
    print("載入模型中...")
    model, preprocess, tokenizer, device = load_model(args.model, args.pretrained)
    labels = load_labels(args.labels_file)
    print(f"使用 {len(labels)} 個標籤進行分類")

    # 執行分類
    results = classify_all(
        args.input_dir, model, preprocess, tokenizer, device, labels, args.threshold,
        extract_location=args.extract_location
    )

    # 印出摘要
    print_summary(results)

    # 儲存結果
    if args.output:
        if args.format == "csv":
            save_csv(results, args.output, labels)
        else:
            save_json(results, args.output)

    # 整理照片
    if args.organize:
        organize_dir = args.organize_dir or (args.input_dir / "organized")
        organize_photos(results, args.input_dir, organize_dir, copy=not args.move)


if __name__ == "__main__":
    main()
