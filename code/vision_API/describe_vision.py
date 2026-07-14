from google.cloud import vision
from google.api_core import exceptions as g_exceptions
from google.auth.exceptions import TransportError

import io
import json
import os
import time


# ========= 1. 初始化 Vision API =========
client = vision.ImageAnnotatorClient.from_service_account_file(
    "socialmediaanalysis-477306-502b5950f40c.json"
)


# ========= 2. 網路錯誤重試函式 =========
def annotate_with_retry(client, image, features, max_backoff: int = 300):
    """
    Vision API 呼叫失敗（網路/503/timeout）時無限重試。
    max_backoff：指數退避最大等待秒數。
    """
    backoff = 5
    while True:
        try:
            return client.annotate_image({'image': image, 'features': features})

        except (
            g_exceptions.ServiceUnavailable,
            g_exceptions.DeadlineExceeded,
            g_exceptions.Unknown,
            g_exceptions.InternalServerError,
            TransportError,
        ) as e:
            print(f"[WARN] API 暫時錯誤：{e}")
            print(f"       {backoff} 秒後重試，同一張圖片不會被跳過...")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

        except Exception as e:
            print(f"[ERROR] 非網路相關錯誤：{e}")
            raise



# ========= 3. 定義偵測 features =========
features = [
    # vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION), # 文字偵測 (檢測圖像中的文字，通常用於簡短文字)
    # vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION), # 文件文字偵測 (專為文件設計，提供更詳細的文字結構資訊)
    # vision.Feature(type_=vision.Feature.Type.LANDMARK_DETECTION), # 地標偵測 (檢測圖像中的著名地標)
    # vision.Feature(type_=vision.Feature.Type.LOGO_DETECTION), # 商標偵測 (檢測圖像中的商標)
    vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION), # 標籤偵測 (檢測圖像的通用標籤或分類)
    vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES), # 圖片屬性偵測 (檢測圖片的屬性，如主色調)
    # vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION), # 物件偵測 (檢測並定位圖像中的特定物件及其邊界框)
    # vision.Feature(type_=vision.Feature.Type.CROP_HINTS), # 裁切建議 (提供最佳的圖片裁切建議及其邊界框和信心分數)
    # vision.Feature(type_=vision.Feature.Type.WEB_DETECTION), # 網路偵測 (在網路上尋找與圖像相關的資訊，如匹配圖片和實體)
    # vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION), # 安全搜尋偵測 (評估圖像是否包含不安全內容，如成人、暴力等)
    # vision.Feature(type_=vision.Feature.Type.FACE_DETECTION) # 臉部偵測 (檢測圖像中的臉部、情緒、地標等)
]


# ========= 4. 遍歷 vision_data 內所有子資料夾 =========
root_folder = "data\\vision_data"

for dirpath, dirnames, filenames in os.walk(root_folder):

    if dirpath == root_folder:
        continue  # 跳過 root（只處理子資料夾）

    output_folder = os.path.join(dirpath, "vision_description")
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n=== 處理資料夾：{dirpath} ===")
    print(f"輸出位置：{output_folder}")

    for filename in filenames:

        # 只處理圖片格式
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue

        image_path = os.path.join(dirpath, filename)

        # ====== ✔ 檢查是否已處理 ======
        output_json_name = os.path.splitext(filename)[0] + "_vision_results.json"
        output_json_path = os.path.join(output_folder, output_json_name)

        if os.path.exists(output_json_path):
            print(f"✔ 已處理，跳過：{filename}")
            continue

        # ====== ↘ 未處理 → 開始分析 ======
        print(f"\n➡ 處理圖片：{filename}")

        try:
            with io.open(image_path, 'rb') as f:
                content = f.read()

            image = vision.Image(content=content)

            # 🔥 使用重試包裝的 API
            response = annotate_with_retry(client, image, features)

            # ======= 你原本超完整的解析邏輯 =======
            # （這段保留你自己的，不改動）
            results_dict = {} # 初始化一個空字典來儲存結果
            results_dict['image_name'] = filename # 將圖片原始檔名加入字典

            # 根據 features 列表中定義的順序，逐一處理並將結果添加到 results_dict 中
            for feature in features:
                if feature.type_ == vision.Feature.Type.TEXT_DETECTION:
                    # 處理文字偵測結果
                    if response.text_annotations:
                        # text_annotations 列表中的第一個項目通常是整個圖像的文字
                        # 我們將每個偵測到的文字區塊的詳細資訊 (description, bounding box, score) 加入列表
                        results_dict['text_annotations'] = []
                        for text in response.text_annotations:
                            vertices = []
                            if text.bounding_poly: # Check if bounding_poly exists
                                for vertex in text.bounding_poly.vertices:
                                    vertices.append({'x': vertex.x, 'y': vertex.y})
                            results_dict['text_annotations'].append({
                                'description': text.description,
                                'bounding_poly': {'vertices': vertices},
                                'score': text.score if hasattr(text, 'score') else 0.0 # Include score if available (not always for text detection), default to 0.0 if none
                            })
                    else:
                        results_dict['text_annotations'] = "No text detected."
                elif feature.type_ == vision.Feature.Type.DOCUMENT_TEXT_DETECTION:
                    # 處理文件文字偵測結果
                    if response.full_text_annotation:
                        # full_text_annotation 包含文件中的完整文字內容及更詳細的結構資訊
                        document_info = {
                            'text': response.full_text_annotation.text,
                            'pages': []
                        }
                        for page in response.full_text_annotation.pages:
                            page_info = {
                                'property': {
                                    'detected_languages': [{'language_code': lang.language_code, 'confidence': lang.confidence} for lang in page.property.detected_languages] if page.property and page.property.detected_languages else [],
                                },
                                'width': page.width,
                                'height': page.height,
                                'blocks': []
                            }
                            for block in page.blocks:
                                block_info = {
                                    'property': {
                                        'detected_languages': [{'language_code': lang.language_code, 'confidence': lang.confidence} for lang in block.property.detected_languages] if block.property and block.property.detected_languages else [],
                                        'detected_break': {'type': block.property.detected_break.type_.name, 'is_prefix': block.property.detected_break.is_prefix} if block.property and block.property.detected_break else None
                                    },
                                    'bounding_box': [{'x': vertex.x, 'y': vertex.y} for vertex in block.bounding_box.vertices],
                                    'paragraphs': [],
                                    'block_type': block.block_type.name,
                                    'confidence': block.confidence
                                }
                                for paragraph in block.paragraphs:
                                    paragraph_info = {
                                        'property': {
                                            'detected_languages': [{'language_code': lang.language_code, 'confidence': lang.confidence} for lang in paragraph.property.detected_languages] if paragraph.property and paragraph.property.detected_languages else [],
                                            'detected_break': {'type': paragraph.property.detected_break.type_.name, 'is_prefix': paragraph.property.detected_break.is_prefix} if paragraph.property and paragraph.property.detected_break else None
                                        },
                                        'bounding_box': [{'x': vertex.x, 'y': vertex.y} for vertex in paragraph.bounding_box.vertices],
                                        'words': [],
                                        'confidence': paragraph.confidence
                                    }
                                    for word in paragraph.words:
                                        word_info = {
                                            'property': {
                                                'detected_languages': [{'language_code': lang.language_code, 'confidence': lang.confidence} for lang in word.property.detected_languages] if word.property and word.property.detected_languages else [],
                                                'detected_break': {'type': word.property.detected_break.type_.name, 'is_prefix': word.property.detected_break.is_prefix} if word.property and word.property.detected_break else None
                                            },
                                            'bounding_box': [{'x': vertex.x, 'y': vertex.y} for vertex in word.bounding_box.vertices],
                                            'symbols': [{'property': {'detected_languages': [{'language_code': lang.language_code, 'confidence': lang.confidence} for lang in symbol.property.detected_languages] if symbol.property and symbol.property.detected_languages else [], 'detected_break': {'type': symbol.property.detected_break.type_.name, 'is_prefix': symbol.property.detected_break.is_prefix} if symbol.property and symbol.property.detected_break else None}, 'bounding_box': [{'x': vertex.x, 'y': vertex.y} for vertex in symbol.bounding_box.vertices], 'text': symbol.text, 'confidence': symbol.confidence} for symbol in word.symbols],
                                            'confidence': word.confidence
                                        }
                                        paragraph_info['words'].append(word_info)
                                    block_info['paragraphs'].append(paragraph_info)
                                page_info['blocks'].append(block_info)
                            document_info['pages'].append(page_info)
                        results_dict['document_text_annotation'] = document_info
                    else:
                        results_dict['document_text_annotation'] = "No document text detected."
                elif feature.type_ == vision.Feature.Type.LANDMARK_DETECTION:
                    # 處理地標偵測結果
                    if response.landmark_annotations:
                        # 遍歷每個偵測到的地標，提取描述、分數和位置資訊
                        results_dict['landmarks'] = []
                        for landmark in response.landmark_annotations:
                            locations = []
                            if landmark.locations:
                                for loc in landmark.locations:
                                    lat_lng = loc.lat_lng
                                    locations.append({'latitude': lat_lng.latitude, 'longitude': lat_lng.longitude})
                            results_dict['landmarks'].append({
                                'description': landmark.description,
                                'score': landmark.score,
                                'locations': locations,
                                'bounding_poly': [{'x': vertex.x, 'y': vertex.y} for vertex in landmark.bounding_poly.vertices] if landmark.bounding_poly else []
                            })
                    else:
                        results_dict['landmarks'] = "No landmarks detected."
                elif feature.type_ == vision.Feature.Type.LOGO_DETECTION:
                    # 處理商標偵測結果
                    if response.logo_annotations:
                        # 遍歷每個偵測到的商標，提取描述、分數和邊界框
                        results_dict['logos'] = []
                        for logo in response.logo_annotations:
                            results_dict['logos'].append({
                                'description': logo.description,
                                'score': logo.score,
                                'bounding_poly': [{'x': vertex.x, 'y': vertex.y} for vertex in logo.bounding_poly.vertices] if logo.bounding_poly else []
                            })
                    else:
                        results_dict['logos'] = "No logos detected."
                elif feature.type_ == vision.Feature.Type.LABEL_DETECTION:
                    # 處理標籤偵測結果 (通用標籤)
                    if response.label_annotations:
                        # 遍歷每個偵測到的標籤，提取描述和分數
                        results_dict['labels'] = [{'description': label.description, 'score': label.score} for label in response.label_annotations]
                    else:
                        results_dict['labels'] = "No labels detected."
                elif feature.type_ == vision.Feature.Type.IMAGE_PROPERTIES:
                    # 處理圖片屬性偵測結果 (主色調)
                    if response.image_properties_annotation:
                        props = response.image_properties_annotation.dominant_colors
                        if props and props.colors:
                            results_dict['image_properties'] = {'dominant_colors': []}
                            # 遍歷每個主色調，提取顏色 (RGB 和 Alpha) 和分數
                            for color_info in props.colors:
                                color = color_info.color

                                # 🔸 把 FloatValue 轉成一般的 float（或 None）
                                alpha_raw = getattr(color, "alpha", None)
                                if hasattr(alpha_raw, "value"):       # FloatValue 物件
                                    alpha_value = float(alpha_raw.value)
                                else:
                                    # 可能是 float、0 或 None
                                    alpha_value = float(alpha_raw) if alpha_raw is not None else None

                                results_dict['image_properties']['dominant_colors'].append({
                                    'color': {
                                        'red': color.red,
                                        'green': color.green,
                                        'blue': color.blue,
                                        'alpha': alpha_value,
                                    },
                                    'score': color_info.score,
                                    'pixel_fraction': color_info.pixel_fraction
                                })
                        else:
                            results_dict['image_properties'] = "No dominant colors detected."
                    else:
                        results_dict['image_properties'] = "No image properties detected."

                elif feature.type_ == vision.Feature.Type.OBJECT_LOCALIZATION:
                    # 處理物件偵測結果 (包括邊界框)
                    if response.localized_object_annotations:
                        results_dict['object_localizations'] = []
                        # 遍歷每個偵測到的物件，提取名稱、分數和邊界框的標準化座標
                        for obj in response.localized_object_annotations:
                            vertices = []
                            for vertex in obj.bounding_poly.normalized_vertices:
                                vertices.append({'x': vertex.x, 'y': vertex.y})
                            results_dict['object_localizations'].append({
                                'name': obj.name,
                                'score': obj.score,
                                'bounding_poly': {'normalized_vertices': vertices}
                            })
                    else:
                        results_dict['object_localizations'] = "No objects detected."
                elif feature.type_ == vision.Feature.Type.CROP_HINTS:
                    # 處理裁切建議 (包括邊界框和信心分數)
                    if response.crop_hints_annotation and response.crop_hints_annotation.crop_hints:
                        results_dict['crop_hints'] = []
                        # 遍歷每個裁切建議，提取邊界框的像素座標和信心分數
                        for crop_hint in response.crop_hints_annotation.crop_hints:
                            vertices = []
                            for vertex in crop_hint.bounding_poly.vertices:
                                vertices.append({'x': vertex.x, 'y': vertex.y})
                            results_dict['crop_hints'].append({
                                'bounding_poly': {'vertices': vertices},
                                'confidence': crop_hint.confidence # 裁切建議的信心分數
                            })
                    else:
                        results_dict['crop_hints'] = "No crop hints detected."
                elif feature.type_ == vision.Feature.Type.WEB_DETECTION:
                    # 處理網路偵測結果
                    if response.web_detection:
                        web_info = {}
                        # 提取網路實體
                        if response.web_detection.web_entities:
                            web_info['web_entities'] = [{'description': entity.description, 'score': entity.score} for entity in response.web_detection.web_entities]
                        # 提取完整匹配圖片
                        if response.web_detection.full_matching_images:
                            web_info['full_matching_images'] = [{'url': image.url} for image in response.web_detection.full_matching_images]
                        # 提取部分匹配圖片
                        if response.web_detection.partial_matching_images:
                            web_info['partial_matching_images'] = [{'url': image.url} for image in response.web_detection.partial_matching_images]
                        # 提取視覺相似圖片
                        if response.web_detection.visually_similar_images:
                            web_info['visually_similar_images'] = [{'url': image.url} for image in response.web_detection.visually_similar_images]
                        # 提取包含匹配圖片的網頁
                        if response.web_detection.pages_with_matching_images:
                            web_info['pages_with_matching_images'] = [{'url': page.url} for page in response.web_detection.pages_with_matching_images]
                        # 提取最佳猜測標籤
                        if response.web_detection.best_guess_labels:
                            web_info['best_guess_labels'] = [{'label': label.label, 'language_code': label.language_code} for label in response.web_detection.best_guess_labels]

                        if web_info:
                            results_dict['web_detection'] = web_info
                        else:
                            results_dict['web_detection'] = "No web detection results."
                    else:
                        results_dict['web_detection'] = "No web detection results."

                elif feature.type_ == vision.Feature.Type.SAFE_SEARCH_DETECTION:
                    # 處理安全搜尋偵測結果
                    if response.safe_search_annotation:
                        safe = response.safe_search_annotation
                        # 將各項安全評估結果 (enum) 轉換為其名稱字串
                        results_dict['safe_search_properties'] = {
                            'adult': safe.adult.name,
                            'spoof': safe.spoof.name,
                            'medical': safe.medical.name,
                            'violence': safe.violence.name,
                            'racy': safe.racy.name
                        }
                    else:
                        results_dict['safe_search_properties'] = "No safe search properties detected."
                elif feature.type_ == vision.Feature.Type.FACE_DETECTION:
                    # 處理臉部偵測結果
                    if response.face_annotations:
                        results_dict['faces'] = []
                        for face in response.face_annotations:
                            # 先選用 fd_bounding_poly，若沒有則退回 bounding_poly
                            box_vertices = []
                            poly = None
                            if getattr(face, "fd_bounding_poly", None) and face.fd_bounding_poly.vertices:
                                poly = face.fd_bounding_poly
                            elif getattr(face, "bounding_poly", None) and face.bounding_poly.vertices:
                                poly = face.bounding_poly

                            if poly is not None:
                                for vertex in poly.vertices:
                                    box_vertices.append({"x": vertex.x, "y": vertex.y})

                            face_info = {
                                "detection_confidence": face.detection_confidence,
                                "joy_likelihood": face.joy_likelihood.name,
                                "sorrow_likelihood": face.sorrow_likelihood.name,
                                "anger_likelihood": face.anger_likelihood.name,
                                "surprise_likelihood": face.surprise_likelihood.name,
                                "under_exposed_likelihood": face.under_exposed_likelihood.name,
                                "blurred_likelihood": face.blurred_likelihood.name,
                                "headwear_likelihood": face.headwear_likelihood.name,
                                "detection_bounding_box": box_vertices,
                                "landmarks": [
                                    {
                                        "type": landmark.type_.name,
                                        "position": {
                                            "x": landmark.position.x,
                                            "y": landmark.position.y,
                                            "z": landmark.position.z,
                                        },
                                    }
                                    for landmark in (face.landmarks or [])
                                ] if face.landmarks else "No landmarks",
                            }
                            results_dict["faces"].append(face_info)
                    else:
                        results_dict["faces"] = "No faces detected."

            # ================================


            # 輸出 JSON
            output_filename = os.path.splitext(filename)[0] + "_vision_results.json"
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=4)

            print(f"✔ 偵測完成 → {output_path}")

        except Exception as e:
            print(f"[ERROR] 處理失敗（不跳過，請修正問題）：{e}")
            raise  # 停下來讓你看錯誤（避免跳過圖片）
