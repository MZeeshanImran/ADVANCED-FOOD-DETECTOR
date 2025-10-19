# ==========================================================
# ADVANCED FOOD DETECTOR + CATEGORY TABLE BILL GENERATOR
# GEMINI 2.5 + YOLO FALLBACK (Pakistan PKR)
# Categorized + Grouped Table Version
# ==========================================================

!pip install -q -U ultralytics Pillow google-generativeai

import io, requests, traceback, random
from PIL import Image
from IPython.display import display, HTML
import google.generativeai as genai
from google.generativeai import GenerativeModel

# ==== USER SETTINGS ====
#image_url = "https://www.bitesbee.com/wp-content/uploads/2021/09/banner-3.jpg"
image_url = "https://www.bitesbee.com/wp-content/uploads/2025/04/LIST-OF-FAST-FOOD-ITEMS-1.jpg"
#image_url = "https://img.freepik.com/free-vector/set-different-foods_1308-27407.jpg?semt=ais_hybrid&w=740&q=80"

# ==== API KEY INPUT ====
from getpass import getpass
api_key = getpass("Enter your Gemini API key: ").strip()
if not api_key:
    raise RuntimeError("❌ Please enter a valid Gemini API key from https://aistudio.google.com/app/apikey")

genai.configure(api_key=api_key)

# ==== DOWNLOAD IMAGE ====
print("📥 Downloading image...")
resp = requests.get(image_url, timeout=20)
resp.raise_for_status()
img = Image.open(io.BytesIO(resp.content)).convert("RGB")
print("✅ Image downloaded:", img.size)
display(HTML("<h4>Original Image:</h4>"))
display(img)

# ==== GEMINI FOOD DETECTION ====
print("\n🔍 Detecting visible food items using Gemini 2.5...\n")

try:
    model = GenerativeModel("models/gemini-2.5-flash")
    prompt = """
You are a food recognition assistant.
List all the visible food items in the provided image.
Be descriptive (e.g., 'chicken burger', 'fries', 'biryani', 'salad bowl').
Return only a clean numbered list. No prices or commentary.
"""
    response = model.generate_content([
        prompt,
        {"mime_type": "image/jpeg", "data": io.BytesIO(resp.content).getvalue()}
    ])
    text = response.text.strip()

    if not text:
        raise RuntimeError("Gemini returned no response.")

    # Extract clean, unique items
    food_items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "." in line:
            line = line.split(".", 1)[-1].strip()
        if line and line not in food_items:
            food_items.append(line)

    if not food_items:
        raise RuntimeError("No food items recognized.")

    print("✅ Gemini successfully recognized the following food items:\n")
    for i, f in enumerate(food_items, 1):
        print(f"{i}. {f}")

except Exception as e:
    print("⚠ Gemini detection failed. Falling back to YOLO (limited classes)...")
    traceback.print_exc()
    from ultralytics import YOLO
    model_yolo = YOLO("yolov8n.pt")
    results = model_yolo.predict(source=img, conf=0.25, imgsz=640, verbose=False)
    det = results[0]
    coco_classes = model_yolo.names
    food_items = []
    if det.boxes is not None and len(det.boxes) > 0:
        class_idxs = det.boxes.cls.cpu().numpy().astype(int)
        for cls_idx in class_idxs:
            label = coco_classes[int(cls_idx)]
            if label in ["pizza","cake","hot dog","donut","sandwich","apple","banana"]:
                if label not in food_items:
                    food_items.append(label)
    if not food_items:
        food_items = ["pizza"]
    print("\n✅ YOLO detected fallback items:")
    for f in food_items:
        print(" -", f)

# ==== PRICE ASSIGNMENT ====
prices = {item: random.randint(350, 950) for item in food_items}

# ==== CATEGORY FUNCTION ====
def categorize(item):
    item_l = item.lower()
    if any(w in item_l for w in ["pasta", "biryani", "curry", "rice", "steak"]):
        return "🍝 Main Course"
    elif any(w in item_l for w in ["burger", "fries", "pizza", "sandwich", "nugget", "wrap", "roll"]):
        return "🍔 Fast Food"
    elif any(w in item_l for w in ["tomato", "lettuce", "salad", "olive", "eggplant", "veggie"]):
        return "🥗 Veggies"
    elif any(w in item_l for w in ["chicken", "meat", "beef", "fish", "mutton"]):
        return "🍗 Meat"
    elif any(w in item_l for w in ["ketchup", "mayo", "sauce", "dip"]):
        return "🍶 Sauces/Dips"
    elif any(w in item_l for w in ["bread", "croissant", "toast", "bun"]):
        return "🥐 Bakery"
    else:
        return "🍽️ Other"

# ==== GROUP ITEMS BY CATEGORY ====
grouped = {}
for item in food_items:
    cat = categorize(item)
    grouped.setdefault(cat, []).append(item)

# ==== DISPLAY CATEGORY-WISE TABLE ====
table_html = """
<h3>🍽️ Available Food Items by Category</h3>
<table border='1' cellpadding='6' style='border-collapse:collapse;text-align:center;'>
<tr><th>S.No</th><th>Category</th><th>Item</th><th>Price (PKR)</th><th>Quantity</th></tr>
"""

counter = 1
# Sort categories in desired order
category_order = ["🍝 Main Course", "🍔 Fast Food", "🍗 Meat", "🥗 Veggies", "🍶 Sauces/Dips", "🥐 Bakery", "🍽️ Other"]

for cat in category_order:
    if cat in grouped:
        table_html += f"<tr><td colspan='5' style='background-color:#f0f0f0; font-weight:bold;'>{cat}</td></tr>"
        for item in grouped[cat]:
            table_html += f"<tr><td>{counter}</td><td>{cat}</td><td>{item}</td><td>{prices[item]}</td><td></td></tr>"
            counter += 1

table_html += "<tr><td colspan='5' style='font-style:italic;'>👉 Enter quantities for selected items below</td></tr>"
table_html += "</table>"
display(HTML(table_html))

# ==== USER SELECTION ====
choices = input("\nEnter the numbers of items you want to order (comma-separated): ").strip()
selected_indices = [int(x)-1 for x in choices.split(",") if x.strip().isdigit() and 0 < int(x) <= len(food_items)]
selected_items = [food_items[i] for i in selected_indices]

# ==== ASK QUANTITY ====
quantities = {}
for item in selected_items:
    while True:
        try:
            q = int(input(f"Enter quantity for '{item}' (default 1): ") or 1)
            if q > 0:
                quantities[item] = q
                break
            else:
                print("❌ Quantity must be at least 1.")
        except:
            print("❌ Please enter a valid number.")

# ==== BILL GENERATION ====
bill_html = """
<h3>🧾 Your Detailed Bill</h3>
<table border='1' cellpadding='6' style='border-collapse:collapse;text-align:center;'>
<tr><th>Item</th><th>Category</th><th>Qty</th><th>Price (PKR)</th><th>Subtotal (PKR)</th></tr>
"""
total = 0
for item, qty in quantities.items():
    subtotal = prices[item] * qty
    total += subtotal
    bill_html += f"<tr><td>{item}</td><td>{categorize(item)}</td><td>{qty}</td><td>{prices[item]}</td><td>{subtotal}</td></tr>"

bill_html += f"<tr><th colspan='4'>Grand Total</th><th>{total}</th></tr></table>"
display(HTML(bill_html))

print("\n✅ Detailed bill generated successfully.")
