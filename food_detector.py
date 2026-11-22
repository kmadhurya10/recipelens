# ═══════════════════════════════════════════════════════════════
#  food_detector.py  —  100% LOCAL food vs non-food classifier
#  Uses CLIP (OpenAI) via transformers — NO API KEY NEEDED
#  First run downloads model (~600MB). After that: fully offline.
#  Food101 REMOVED — CLIP only with rich visual descriptors
#  Works better for Indian food, bowls, vessels, all angles
# ═══════════════════════════════════════════════════════════════

import os
import torch
from PIL import Image

# ═══════════════════════════════════════════════════════════════
#  CLIP — Lazy Load
# ═══════════════════════════════════════════════════════════════

_clip_model     = None
_clip_processor = None
_model_loaded   = False
_load_error     = None

def _load_clip():
    global _clip_model, _clip_processor, _model_loaded, _load_error
    if _model_loaded:
        return True
    if _load_error:
        return False
    try:
        from transformers import CLIPProcessor, CLIPModel
        print("[FoodDetector] Loading CLIP model (first run downloads ~600MB)...")
        _clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.eval()
        _model_loaded = True
        print("[FoodDetector] CLIP model loaded successfully!")
        return True
    except Exception as e:
        _load_error = str(e)
        print(f"[FoodDetector] CLIP load failed: {e}")
        return False

# Dummy food101 variables so app.py status route does not break
_food101_loaded = False
_food101_error  = "Food101 disabled — using CLIP only"


# ═══════════════════════════════════════════════════════════════
#  CLIP TEXT LABELS — food vs non-food
# ═══════════════════════════════════════════════════════════════

FOOD_LABELS = [
    "a photo of food",
    "a photo of a meal",
    "a photo of a dish",
    "a photo of cooked food",
    "a photo of a recipe",
    "a photo of vegetables",
    "a photo of fruits",
    "a photo of bread or baked goods",
    "a photo of a dessert or sweet",
    "a photo of a drink or beverage",
    "a photo of raw ingredients",
    "a photo of a restaurant meal",
    "a photo of street food",
    "a photo of rice or biryani",
    "a photo of curry or soup in a bowl",
    "a photo of a sandwich or burger",
    "a photo of pizza",
    "a photo of sushi or Japanese food",
    "a photo of Indian food in a bowl or plate",
    "a photo of Chinese food",
    "a photo of breakfast food",
    "a photo of snacks",
    "a photo of seafood",
    "a photo of meat or chicken",
    "a photo of a salad",
    "a photo of food served in a steel or copper vessel",
    "a photo of food served in a bowl",
    "a photo of Indian sweets or mithai",
    "a photo of biryani or pulao rice",
    "a photo of dal or lentil curry",
]

NON_FOOD_LABELS = [
    "a photo of a person or human face",
    "a photo of an animal or pet",
    "a photo of a car or vehicle",
    "a photo of a building or house",
    "a photo of nature or landscape",
    "a photo of a tree or flower",
    "a photo of a phone or laptop",
    "a photo of a book or document",
    "a photo of furniture or room interior",
    "a photo of clothing or fashion",
    "a photo of money or wallet",
    "a photo of art or drawing",
    "a photo of a map or chart",
    "a photo of a pen or stationery",
    "a photo of sports equipment",
    "a photo of a toy or game",
    "a photo of a tool or machine",
    "a photo of a street or road",
    "a photo of the sky or clouds",
    "a photo of a mountain or beach",
    "a photo of dfd diagram or screenshot",
]

ALL_LABELS = FOOD_LABELS + NON_FOOD_LABELS
FOOD_COUNT = len(FOOD_LABELS)

# Lowered threshold so bowl/vessel photos pass more easily
FOOD_THRESHOLD = 0.40


# ═══════════════════════════════════════════════════════════════
#  VISUAL DESCRIPTORS — rich descriptions for accurate matching
# ═══════════════════════════════════════════════════════════════

VISUAL_DESCRIPTORS = {
    # ── Rice dishes ────────────────────────────────────────────
    "mutton biryani": (
        "mutton biryani, Indian fragrant basmati rice cooked with large bone-in mutton pieces, "
        "saffron yellow and brown colored rice served in a copper or steel handi vessel, "
        "garnished with fried onions mint and coriander, whole spices visible, NOT Chinese fried rice"
    ),
    "hyderabadi biryani": (
        "Hyderabadi biryani, layered basmati rice with whole spices saffron color, "
        "large pieces of mutton or chicken visible, served in a large handi vessel, "
        "South Indian Hyderabadi style biryani"
    ),
    "biryani": (
        "biryani, fragrant saffron yellow and brown basmati rice mixed with meat and whole spices, "
        "garnished with fried onions and coriander leaves, served in a bowl or vessel, "
        "Indian biryani NOT Chinese fried rice"
    ),
    "pulao": (
        "vegetable pulao, mildly spiced white basmati rice cooked with vegetables, "
        "lighter colored rice dish compared to biryani, no saffron color"
    ),
    "curd rice": (
        "curd rice, South Indian white rice mixed with yogurt curd and tempering of mustard seeds, "
        "white creamy rice dish with curry leaves served in a bowl"
    ),
    "fried rice": (
        "Chinese fried rice, stir fried rice with eggs vegetables and soy sauce, "
        "served in a Chinese wok or bowl, NOT Indian biryani"
    ),
    

    # ── Flatbreads ─────────────────────────────────────────────
    "aloo paratha": (
        "aloo paratha, Indian whole wheat flatbread stuffed with spiced mashed potato filling, "
        "served with white butter and yogurt curd, North Indian Punjabi food"
    ),
    "holige": (
        "holige puran poli, Karnataka sweet flatbread stuffed with yellow chana dal and jaggery, "
        "thin golden round sweet stuffed bread served with ghee, South Indian dessert"
    ),
    "puran poli": (
        "puran poli, Maharashtrian sweet flatbread stuffed with chana dal jaggery filling, "
        "thin golden round sweet stuffed bread served with ghee, Indian festive dessert"
    ),
    "roti": (
        "plain roti chapati, simple thin whole wheat flatbread cooked on tawa, "
        "no stuffing, dry Indian bread, served on a plate"
    ),
    "naan": (
        "naan bread, leavened Indian flatbread baked in tandoor oven, "
        "fluffy soft white bread with charred spots, served in Indian restaurants"
    ),
    "paratha": (
        "plain paratha, layered flaky whole wheat flatbread cooked with oil or ghee on tawa, "
        "no potato stuffing, multi-layered Indian flatbread"
    ),
    
    "paneer lababdar": (
        "paneer lababdar, creamy orange-red tomato onion gravy with white paneer cubes, "
        "thick rich North Indian curry with butter and cream swirl, served in a bowl"
    ),
    "aloo dum biryani": (
        "aloo dum biryani, fragrant saffron basmati rice layered with whole spiced potatoes, "
        "fried onion birista on top, served in a deep vessel, Kolkata style vegetarian biryani"
    ),
    "sindhi kadhi": (
        "Sindhi kadhi, thick yellow-brown besan lentil curry with vegetables like drumstick and potato, "
        "tamarind sour flavor, thicker than regular kadhi, served in a bowl with rice"
    ),
    "sev tamatar sabzi": (
        "sev tamatar sabzi, thin crispy yellow sev chickpea noodles in red tomato gravy, "
        "Gujarati or Rajasthani dry curry with crunchy sev strands, served in a plate"
    ),
    "batata nu shaak": (
        "batata nu shaak, Gujarati dry potato curry, golden cubed potatoes tossed in mustard "
        "seeds curry leaves and turmeric, no gravy, dry preparation in a bowl"
    ),
    "maharashtrian amti": (
        "amti, thin Maharashtrian lentil curry with jaggery and tamarind, golden yellow dal "
        "with sweet sour flavor, thin consistency served in a bowl with rice"
    ),
    "sol kadi": (
        "sol kadi, bright pink-purple cooling drink made from kokum and coconut milk, "
        "served in a glass, Konkan Goan digestive beverage, deep rose colored liquid"
    ),
    "thalipeeth": (
        "thalipeeth, thick multigrain Maharashtrian flatbread made from jowar bajra besan rice flour, "
        "dark speckled flatbread cooked with oil, served on a plate with yogurt"
    ),
    "mirchi vada": (
        "mirchi vada, large green banana pepper stuffed with spiced potato, "
        "coated in thick yellow besan batter and deep fried, Rajasthani street food on a plate"
    ),
    "methi matar malai": (
        "methi matar malai, white creamy gravy with dark green methi fenugreek leaves "
        "and green peas, mild aromatic North Indian curry served in a bowl"
    ),
    "cholar dal": (
        "cholar dal, Bengali split Bengal gram lentil with coconut pieces and raisins, "
        "golden yellow dal with white coconut pieces visible, served in a bowl"
    ),
    "mughlai paratha": (
        "Mughlai paratha, flaky all-purpose flour paratha with egg filling inside, "
        "golden brown layered flatbread with egg visible at edges, Kolkata street food"
    ),
    "koraishutir kochuri": (
        "koraishutir kochuri, golden deep fried puffed round puri stuffed with spiced green pea filling, "
        "Bengali green pea stuffed puri on a plate, fluffy golden fried bread"
    ),
    "chana jor garam": (
        "chana jor garam, dark brown spiced dry black chickpea street food snack, "
        "served in a paper cone or bowl with onion tomato and sev on top, Indian chaat"
    ),
    "shakarpara": (
        "shakarpara, small golden diamond or square shaped deep fried sweet cookies, "
        "crispy dry Indian festival snack, light golden color arranged on a plate"
    ),
    "mawa kachori": (
        "mawa kachori, round deep fried pastry dipped in sugar syrup with khoya mawa filling, "
        "golden brown glistening with syrup, Rajasthani sweet kachori on a plate"
    ),
    "paneer makhani": (
        "paneer makhani, smooth creamy orange-red butter tomato gravy with large white paneer cubes, "
        "restaurant style North Indian curry, similar to butter chicken but vegetarian, rich sauce"
    ),
    "palak dal": (
        "palak dal, yellow lentil cooked with spinach, light green colored thick dal "
        "with spinach visible, served in a bowl with ghee drizzle"
    ),
    "lasooni dal": (
        "lasooni dal, yellow toor dal with golden fried garlic tempering on top, "
        "hot garlic tadka sizzling over the dal, served in an earthen bowl"
    ),
    "chana dal tadka": (
        "chana dal, yellow split Bengal gram lentil with cumin and chili tempering, "
        "slightly thick textured dal served in a bowl, different from moong dal"
    ),
    "moong dal cheela": (
        "moong dal cheela, thin yellow-green savory crepe made from yellow moong lentil batter, "
        "served on a plate with green chutney, similar to dosa but yellow colored"
    ),
    "aloo tikki": (
        "aloo tikki, flat round pan-fried potato patties, golden crispy outside, "
        "served with green chutney and tamarind chutney and yogurt on a plate, North Indian street food"
    ),
    "raj kachori": (
        "raj kachori, large round hollow crispy fried pastry shell filled with yogurt "
        "sev chickpeas and chutneys, Delhi street food chaat topped with colorful toppings"
    ),
    "dahi puri": (
        "dahi puri, small round crispy puri shells filled with yogurt tamarind chutney "
        "sev and spices, Mumbai street food chaat on a plate, white yogurt drizzled over"
    ),
    "papdi chaat": (
        "papdi chaat, crispy fried flour cracker discs topped with yogurt chickpeas tamarind "
        "chutney and sev, colorful North Indian chaat platter"
    ),
    "aloo chaat": (
        "aloo chaat, crispy fried potato cubes tossed with spices tamarind chutney "
        "and green chutney, tangy dry potato snack in a bowl or plate, Delhi street food"
    ),
    "corn chaat": (
        "corn chaat, boiled sweet corn kernels tossed with butter lime chaat masala "
        "and spices, yellow corn in a cup or bowl, Indian street food snack"
    ),
    "katori chaat": (
        "katori chaat, crispy fried flour bowl (katori) filled with chickpeas yogurt "
        "and chutneys, edible bowl filled with chaat toppings, North Indian street food"
    ),
    "tokri chaat": (
        "tokri chaat, crispy potato basket filled with various chaat toppings "
        "like yogurt sev and chutneys, decorative presentation, festive Indian chaat"
    ),
    "aloo bonda": (
        "aloo bonda, round deep fried potato dumpling coated in besan chickpea batter, "
        "golden round fritter similar to vada, South Indian tea time snack on a plate"
    ),
    "punugulu": (
        "punugulu, small round crispy fried rice lentil batter balls, "
        "Andhra street food snack served with coconut chutney, golden brown bite-sized balls"
    ),
    "mangalore bajji": (
        "mangalore bajji, fluffy soft deep fried all-purpose flour fritters, "
        "golden brown pillowy bhajji from coastal Karnataka, served with coconut chutney"
    ),
    "rava idli": (
        "rava idli, white steamed semolina cakes, softer and slightly grainy texture "
        "compared to regular idli, served with sambar and coconut chutney, Karnataka style"
    ),
    "set dosa": (
        "set dosa, thick soft spongy round dosas served in a set of three, "
        "smaller and thicker than regular dosa, Karnataka breakfast, served with chutney"
    ),
    "neer dosa": (
        "neer dosa, very thin white delicate rice crepes, translucent and lacy texture, "
        "rolled into a cone shape, Mangalorean specialty, served with coconut chutney"
    ),
    "akki rotti": (
        "akki rotti, white rice flour flatbread from Karnataka with onion carrot "
        "and curry leaves visible, cooked on tawa until crispy, served with chutney"
    ),
    "ragi mudde": (
        "ragi mudde, dark brown-black round ball made from finger millet (ragi) flour, "
        "very dense heavy ball eaten with sambar, Karnataka village food"
    ),
    "amboli": (
        "amboli, thick white soft fermented rice lentil pancake from Maharashtra, "
        "similar to uttapam but softer, served with chutney and curry"
    ),
    "ghavane": (
        "ghavane, thin white crispy rice crepes from Goa Maharashtra, "
        "similar to neer dosa, translucent white rolled crepes served with curry"
    ),
    "modak": (
        "modak, steamed white rice flour dumpling with coconut jaggery filling inside, "
        "pyramid shape with pinched top, Ganesh festival sweet, served on a plate"
    ),
    "puran poli maharashtrian": (
        "puran poli, thin golden Maharashtra sweet stuffed flatbread with chana dal jaggery filling, "
        "cooked with ghee, round golden bread served with ghee drizzle"
    ),
    "ukdiche modak": (
        "ukdiche modak, steamed white rice flour modak dumplings, "
        "pure white pyramid shaped dumplings arranged on a plate, Ganesh Chaturthi prasad"
    ),
    "tukda": (
        "double ka meetha, deep fried bread slices soaked in saffron sugar syrup "
        "topped with rabri and dry fruits, Hyderabadi bread pudding dessert in a bowl"
    ),
    "qubani ka meetha": (
        "qubani ka meetha, Hyderabadi apricot dessert, soft orange-brown cooked dried apricots "
        "in thick sweet syrup topped with cream, served in a glass or bowl"
    ),
    "sheer khurma": (
        "sheer khurma, creamy white milk based vermicelli dessert with dates "
        "dry fruits and saffron, Eid festival sweet, served in a bowl"
    ),
    "phirni": (
        "phirni, thick creamy white rice flour pudding set in earthen clay bowls, "
        "garnished with pistachios and silver vark, served chilled in individual portions"
    ),
    "basundi": (
        "basundi, thick reduced sweetened milk dessert similar to rabri, "
        "creamy white with layers of cream, garnished with nuts and saffron, Gujarati dessert"
    ),
    "rabri": (
        "rabri, thick layered reduced sweetened milk with visible cream layers, "
        "served in a bowl or clay cup, white and creamy with saffron color"
    ),
    "kulfi": (
        "kulfi, Indian frozen dairy dessert, cone shaped or on a stick, "
        "creamy dense ice cream with pistachio or saffron flavor, garnished with falooda"
    ),
    "falooda": (
        "falooda, tall glass of layered dessert drink with rose syrup vermicelli "
        "basil seeds ice cream and milk, pink colored layered beverage, Mumbai street dessert"
    ),
    "rose milk": (
        "rose milk, bright pink cold milk drink flavored with rose syrup, "
        "served in a tall glass with ice, popular South Indian beverage"
    ),
    "badam milk": (
        "badam milk, thick creamy almond milk flavored with saffron and cardamom, "
        "golden yellow color, served hot or cold in a glass, Indian health drink"
    ),
    "thandai": (
        "thandai, thick creamy spiced milk drink with almonds saffron pepper and rose petals, "
        "white creamy beverage served in a glass, Holi festival drink"
    ),
    "aam panna": (
        "aam panna, bright green cold raw mango drink, tangy and sweet green colored beverage "
        "in a glass with ice, Indian summer cooler"
    ),
    "jaljeera": (
        "jaljeera, dark green spiced cumin water drink in a glass, "
        "Indian digestive cold beverage with mint and spices, served with boondi floating on top"
    ),
    "nimbu pani": (
        "nimbu pani, fresh lemon water with mint in a tall glass, "
        "pale yellow Indian lemonade with ice, simple refreshing summer drink"
    ),
    "coconut water": (
        "fresh coconut water, clear water served inside a green or white coconut, "
        "tropical natural drink, coconut with straw, Indian beach drink"
    ),
    "sugarcane juice": (
        "sugarcane juice, pale yellow-green fresh pressed juice in a glass, "
        "Indian street beverage extracted from sugarcane stalks, served with ice and lemon"
    ),
    "kesar badam kulfi": (
        "kesar badam kulfi, saffron almond Indian frozen dessert, "
        "deep orange-yellow colored dense ice cream on a stick, garnished with pistachios"
    ),
    "besan ladoo": (
        "besan ladoo, round golden yellow ball shaped sweet made from roasted gram flour and ghee, "
        "smooth dense ball, dry surface, Indian festival sweet"
    ),
    "motichoor ladoo": (
        "motichoor ladoo, round orange-yellow ladoo made from tiny fried besan pearls, "
        "soft and moist texture, garnished with pistachios, Diwali and wedding sweet"
    ),
    "boondi ladoo": (
        "boondi ladoo, round sweet ball made from fried besan boondi pearls, "
        "orange-golden color, slightly sticky surface with saffron and cardamom"
    ),
    "til ladoo": (
        "til ladoo, round black and white sesame seed jaggery balls, "
        "dark speckled surface, dense and crispy, Makar Sankranti festival sweet"
    ),
    "peanut chikki": (
        "peanut chikki, hard brittle slab of peanuts set in jaggery, "
        "golden brown transparent jaggery with whole peanuts visible, Indian candy bar"
    ),
    "sesame chikki": (
        "sesame chikki, thin brittle slab of sesame seeds set in jaggery, "
        "dark golden brown with white sesame seeds, cut into rectangles"
    ),
    "khaja": (
        "khaja, flaky deep fried layered pastry soaked in sugar syrup, "
        "golden crispy layered sweet from Odisha, rectangular or round shape"
    ),
    "anarsa": (
        "anarsa, thin crispy rice flour sweet disc coated with white sesame seeds, "
        "Maharashtrian Diwali sweet, light golden disc shape"
    ),
    "chiroti": (
        "chiroti, delicate flaky deep fried pastry dusted with powdered sugar, "
        "Karnataka festive sweet, light golden color with white sugar dusting"
    ),
    "karanji": (
        "karanji, deep fried crescent moon shaped pastry filled with coconut jaggery, "
        "golden crispy half moon shape, Maharashtra Diwali sweet"
    ),
    "chakli": (
        "chakli, crispy spiral shaped deep fried rice flour snack, "
        "golden brown circular coil with ridges, Indian savory Diwali snack"
    ),
    "murukku": (
        "murukku, crispy spiral or twisted deep fried rice flour snack, "
        "golden brown crunchy Indian snack, South Indian chakli with different shape"
    ),
    "ribbon pakoda": (
        "ribbon pakoda, flat crispy ribbon shaped deep fried besan snack, "
        "golden wavy flat strips, Indian savory festival snack"
    ),
    "kara sev": (
        "kara sev, thick crispy fried chickpea flour noodles, "
        "golden yellow cylindrical crispy snack, different from thin sev"
    ),
    "mathri": (
        "mathri, flat round crispy flaky deep fried flour crackers, "
        "golden puffed flat disc, Indian savory snack with ajwain seeds visible"
    ),
    "namak pare": (
        "namak pare, small crispy diamond shaped deep fried flour snacks, "
        "golden savory crackers with sesame or ajwain, Indian festival namkeen"
    ),
    "khakhra": (
        "khakhra, thin crispy toasted wheat flour cracker from Gujarat, "
        "light golden flat dry cracker served on a plate, no oil dry texture"
    ),
    "papad": (
        "papad, thin crispy round disc of lentil or rice flour, "
        "fried or roasted, white or golden with speckled black pepper visible"
    ),
    "fafda": (
        "fafda, long crispy golden strips of fried besan chickpea flour, "
        "flat yellow strips served with papaya chutney, Gujarati breakfast snack"
    ),
    "gathiya": (
        "gathiya, soft thick fried besan noodles, yellow cylindrical soft fried snack, "
        "different from chakli, Gujarati snack served with chutney"
    ),
    "moong dal pakoda": (
        "moong dal pakoda, crispy fried fritters made from soaked yellow moong lentils, "
        "golden rough-textured round fritters, rainy day Indian snack"
    ),
    "pyaz pakoda": (
        "pyaz pakoda, onion fritters, crispy golden brown fried onion rings and strands "
        "in besan batter, Indian monsoon snack served with chutney"
    ),
    "aloo pakoda": (
        "aloo pakoda, thin potato slices coated in spiced besan batter and deep fried, "
        "flat golden crispy potato fritters, Indian tea time snack"
    ),
    "paneer pakoda": (
        "paneer pakoda, thick rectangular paneer pieces dipped in besan batter and deep fried, "
        "golden crispy outer coating with soft white paneer inside"
    ),
    "bread pakoda": (
        "bread pakoda, white bread slices stuffed with spiced potato and deep fried in besan batter, "
        "golden rectangular fried snack, North Indian street food"
    ),
    "corn pakoda": (
        "corn pakoda, fresh corn kernels mixed in spiced besan batter and deep fried, "
        "golden crispy irregular shaped fritters with corn visible inside"
    ),
    "palak pakoda": (
        "palak pakoda, fresh spinach leaves coated in besan batter and deep fried, "
        "flat green-golden leaf fritters, crispy Indian monsoon snack"
    ),
    "maddur vade": (
        "maddur vade, thin crispy flat vada from Karnataka, "
        "golden flat disc with onion and curry leaves visible, highway snack from Maddur"
    ),
    "ambode": (
        "ambode, thick crispy chana dal vada from Karnataka, "
        "rough textured golden brown round fritter, different from medu vada"
    ),
    "masala vada": (
        "masala vada, crispy thick chana dal vada with onion and green chili inside, "
        "rough golden brown flat round fritter, South Indian snack"
    ),
    "sabudana vada": (
        "sabudana vada, crispy round fried patties made from tapioca pearls and potato, "
        "golden white-speckled vada, fasting food on a plate"
    ),
    "maida halwa": (
        "maida halwa, thick glossy dense all-purpose flour pudding, "
        "deep brown or orange color, poured into greased plate and cut into pieces"
    ),
    "atta halwa": (
        "atta halwa, thick dense whole wheat flour pudding cooked in ghee and sugar, "
        "dark brown glossy texture, served in a bowl with ghee"
    ),
    "carrot halwa": (
        "gajar ka halwa, bright orange carrot pudding cooked in milk and ghee, "
        "shredded carrot visible in thick sweet mixture, garnished with nuts"
    ),
    "beetroot halwa": (
        "beetroot halwa, deep red-purple beet root pudding, "
        "vibrant purple-pink colored thick Indian dessert, garnished with nuts"
    ),
    "bottle gourd halwa": (
        "lauki halwa, white or cream colored bottle gourd pudding, "
        "shredded lauki in sweet thick mixture, garnished with saffron and nuts"
    ),
    "suran curry": (
        "suran ki sabzi, yam curry, dark brown or purple colored dry or semi-dry dish "
        "of elephant foot yam tossed in spices, served in a bowl"
    ),
    "arbi masala": (
        "arbi masala, colocasia taro root curry, golden crispy taro pieces in dry masala, "
        "small round or oval pieces coated in spices, served as a side dish"
    ),
    "kathal biryani": (
        "kathal biryani, raw jackfruit biryani, saffron basmati rice layered with "
        "spiced raw jackfruit pieces, looks like meat biryani but vegetarian"
    ),
    "kathal sabzi": (
        "kathal sabzi, raw jackfruit curry, firm fibrous jackfruit pieces in dark spiced gravy, "
        "the texture resembles pulled meat, served in a bowl"
    ),
    "banana flower curry": (
        "vazhaipoo curry, banana blossom flower curry, cream colored fibrous flower "
        "cooked with coconut and spices, South Indian preparation"
    ),
    "drumstick curry": (
        "murungakkai curry, long green drumstick pieces in tomato coconut gravy, "
        "bright green cylinder shaped vegetables in curry, South Indian preparation"
    ),
    "raw jackfruit stir fry": (
        "green jackfruit stir fry, dry spiced raw jackfruit pieces with coconut, "
        "Kerala style, fibrous pale white-green jackfruit cooked dry"
    ),
    "stuffed karela": (
        "bharwan karela, stuffed bitter gourd, dark green wrinkled karela filled "
        "with spiced onion filling, tied with thread, cooked until slightly charred"
    ),
    "kadhi pakoda": (
        "kadhi pakoda, yellow sour yogurt besan curry with fried pakoda dumplings floating inside, "
        "North Indian golden yellow curry with fritters, served in a bowl with rice"
    ),
    "gujarati kadhi": (
        "Gujarati kadhi, thin sweet-sour yellow yogurt curry, more liquid than Punjabi kadhi, "
        "light yellow color served in a bowl, sweet note from jaggery"
    ),
    "rajasthani kadhi": (
        "Rajasthani kadhi, thick yellow besan yogurt curry, no pakodas, "
        "tangy and thick served in a bowl with steamed rice"
    ),
    "watermelon rind curry": (
        "watermelon rind sabzi, white-green watermelon rind cooked in dry spices, "
        "unusual zero-waste Indian vegetable dish, light colored"
    ),
    "raw papaya sabzi": (
        "raw papaya sabzi, green unripe papaya curry, pale green colored dry dish "
        "with mustard seeds and curry leaves, Bengali or South Indian preparation"
    ),
    "ridge gourd curry": (
        "turai curry, ridge gourd curry, dark green ridged vegetable pieces in light curry, "
        "soft green vegetable in tomato onion gravy"
    ),
    "cluster beans sabzi": (
        "gawar phali sabzi, cluster beans dry curry, thin green beans tossed in "
        "peanut coconut spice mixture, Gujarati or Rajasthani style, dry preparation"
    ),
    "suva methi sabzi": (
        "suva methi sabzi, dill and fenugreek leaves stir fry, "
        "dark green fragrant mixed greens cooked dry with spices, Gujarat preparation"
    ),
    "tendli curry": (
        "tendli curry, ivy gourd curry, small oval green vegetables in coconut tomato gravy, "
        "Goa Karnataka curry, tiny green cylinders in spiced gravy"
    ),
    "ambat varan": (
        "ambat varan, Goan sour lentil curry with kokum, "
        "thin yellow dal with pink-red sourness from kokum, served with rice"
    ),
    "pathiri": (
        "pathiri, thin white rice flour flatbread from Kerala Malabar region, "
        "very thin delicate white circle, served with chicken or vegetable curry"
    ),
    "puttu": (
        "puttu, steamed cylindrical rice flour cake with coconut layers, "
        "white cylinder of steamed rice served on a plate, Kerala breakfast"
    ),
    "idiyappam": (
        "idiyappam, steamed rice noodle nest, white delicate string hoppers pressed "
        "into a flat spiral, Kerala Tamil breakfast served with coconut milk"
    ),
    "kozhukattai": (
        "kozhukattai, steamed rice flour dumplings with savory or sweet filling, "
        "white oval or conical shaped steamed dumplings, Tamil Vinayaka Chaturthi sweet"
    ),
    "arisi upma": (
        "arisi upma, thick dry coarse rice upma, grainy white-beige texture, "
        "South Indian broken rice porridge cooked with lentils and spices"
    ),
    "thayir sadam": (
        "thayir sadam, curd rice, white creamy rice mixed with yogurt and "
        "tempered with mustard seeds curry leaves and pomegranate on top, South Indian comfort food"
    ),
    "chitranna": (
        "chitranna, Karnataka lemon rice, bright yellow rice tossed with turmeric "
        "lemon juice peanuts and curry leaves, tangy yellow colored rice"
    ),
    "tamarind rice": (
        "puliyodarai tamarind rice, dark brown tangy rice mixed with tamarind paste "
        "and spices, deep brown colored South Indian rice preparation"
    ),
    "coconut rice": (
        "coconut rice, white rice tossed with grated fresh coconut and mustard seeds, "
        "white rice with visible white coconut flakes, South Indian temple offering"
    ),
    "sesame rice": (
        "sesame rice, ellu sadam, white rice coated with white sesame seeds "
        "and spices, speckled white rice, South Indian preparation"
    ),


    # ── Curries ────────────────────────────────────────────────
    "dal tadka": (
        "dal tadka, yellow lentil soup topped with red chili tempering in ghee, "
        "served in a bowl with oil floating on top, Indian lentil curry"
    ),
    "dal makhani": (
        "dal makhani, creamy dark black urad dal cooked overnight, "
        "dark colored thick lentil curry with cream swirl on top, served in a bowl"
    ),
    "butter chicken": (
        "butter chicken murgh makhani, bright orange-red creamy tomato curry with chicken pieces, "
        "rich smooth North Indian curry sauce served in a bowl"
    ),
    "paneer butter masala": (
        "paneer butter masala, orange-red creamy tomato gravy with white paneer cubes, "
        "North Indian vegetarian curry served in a bowl"
    ),
    "palak paneer": (
        "palak paneer, thick dark green spinach gravy with white paneer cottage cheese cubes, "
        "Indian vegetarian curry served in a bowl"
    ),
    "mutton curry": (
        "mutton curry, dark brown spicy gravy with large bone-in mutton pieces, "
        "thick oily Indian meat curry served in a bowl"
    ),
    "rajma": (
        "rajma, red kidney bean curry in thick tomato onion gravy, "
        "North Indian red bean curry served in a bowl"
    ),

    # ── South Indian ───────────────────────────────────────────
    "dosa": (
        "masala dosa, large thin crispy golden crepe made from fermented rice batter, "
        "folded with potato filling inside, South Indian breakfast served on a plate"
    ),
    "idli": (
        "idli, round white steamed rice cakes served with sambar and coconut chutney, "
        "soft white circular South Indian breakfast on a plate"
    ),
    "vada": (
        "medu vada, round donut-shaped crispy fried lentil fritter with hole in center, "
        "South Indian snack served with chutney"
    ),
    "sambhar": (
        "sambar, thin runny South Indian lentil vegetable soup with tamarind, "
        "dark brownish soup with vegetables served in a bowl"
    ),
    "rasam": (
        "rasam, very thin watery South Indian tamarind pepper soup, "
        "light colored thin liquid South Indian soup served in a bowl"
    ),
    "uttapam": (
        "uttapam, thick soft rice pancake with onion tomato toppings on surface, "
        "thick South Indian pancake with colorful vegetable toppings"
    ),
    "pongal": (
        "pongal, creamy white soft rice and moong dal cooked with ghee and black pepper, "
        "South Indian white creamy rice dish served in a bowl"
    ),

    # ── Snacks / Street food ───────────────────────────────────
    "samosa": (
        "samosa, triangular deep fried pastry stuffed with spiced potato, "
        "crispy golden brown triangle-shaped Indian snack"
    ),
    "pani puri": (
        "pani puri golgappa, small round hollow fried crispy puri balls filled with spiced water, "
        "Indian street food chaat served in a plate"
    ),
    "bhel puri": (
        "bhel puri, Indian street food chaat made of puffed rice mixed with vegetables, "
        "tamarind chutney sev onion tomato served in a plate or cone"
    ),
    "chole bhature": (
        "chole bhature, spicy dark brown chickpea curry served with large deep fried puffy white bread, "
        "North Indian meal with big balloon-shaped bhature bread"
    ),
    "pav bhaji": (
        "pav bhaji, thick spiced mashed vegetable curry served with buttered soft bread rolls, "
        "Mumbai street food red-orange vegetable mash with bread"
    ),
    "misal pav": (
        "misal pav, spicy sprouted moth bean curry topped with farsan sev, "
        "Maharashtrian spicy snack with bread rolls and crunchy toppings"
    ),
    

    # ── Meat dishes ────────────────────────────────────────────
    "hyderabadi haleem": (
        "Hyderabadi haleem, thick porridge-like dish of slow-cooked mutton and broken wheat, "
        "dark brown paste consistency garnished with fried onions lemon ginger served in a bowl"
    ),
    "tandoori chicken": (
        "tandoori chicken, red marinated whole chicken pieces cooked in tandoor oven, "
        "charred red orange grilled chicken served with lemon and onion rings"
    ),
    "chicken tikka": (
        "chicken tikka, small boneless chicken pieces marinated in red spices and grilled, "
        "red charred chicken chunks on skewer or plate"
    ),

    # ── Sweets / Desserts ──────────────────────────────────────
    "gulab jamun": (
        "gulab jamun, small dark brown round fried milk dumplings soaked in sugar syrup, "
        "dark golden brown color NOT white, Indian sweet dessert served in a bowl"
    ),
    "jalebi": (
        "jalebi, bright orange spiral-shaped deep fried sweet soaked in sugar syrup, "
        "crispy coiled orange Indian sweet on a plate"
    ),
    "kheer": (
        "kheer, creamy white rice pudding dessert garnished with nuts and cardamom, "
        "Indian sweet rice porridge served in a bowl"
    ),
    "halwa": (
        "halwa, dense smooth glossy orange or brown Indian sweet pudding made from semolina or carrot, "
        "served in a bowl with ghee"
    ),
    "rasgulla": (
        "rasgulla, soft spongy white perfectly round balls made from chenna cottage cheese "
        "soaked in clear thin transparent sugar syrup, no cream no saffron, "
        "Bengali Indian sweet, watery clear syrup NOT yellow or creamy"
    ),
    "rasmalai": (
        "rasmalai, soft white flattened disc shaped paneer soaked in thick creamy saffron "
        "yellow rabri milk, garnished with pistachios and saffron strands, "
        "flat discs NOT round balls, yellow creamy liquid NOT clear syrup"
    ),
}


# ═══════════════════════════════════════════════════════════════
#  CLIP CATEGORY LABELS — built from recipe database
# ═══════════════════════════════════════════════════════════════

_FOOD_CATEGORY_LABELS = None
_FOOD_CATEGORY_KEYS   = None

def _get_food_category_labels():
    global _FOOD_CATEGORY_LABELS, _FOOD_CATEGORY_KEYS
    if _FOOD_CATEGORY_LABELS is not None:
        return _FOOD_CATEGORY_LABELS, _FOOD_CATEGORY_KEYS

    try:
        from database.food_dataset import RECIPE_DATABASE
    except ImportError:
        _FOOD_CATEGORY_LABELS = []
        _FOOD_CATEGORY_KEYS   = []
        return _FOOD_CATEGORY_LABELS, _FOOD_CATEGORY_KEYS

    labels = []
    keys   = []

    for key, recipe in RECIPE_DATABASE.items():
        food_name    = recipe.get("food_name", key)
        cuisine_type = recipe.get("cuisine_type", "")

        descriptor = VISUAL_DESCRIPTORS.get(key.lower())
        if not descriptor:
            descriptor = VISUAL_DESCRIPTORS.get(food_name.lower())

        if descriptor:
            label = f"a photo of {descriptor}"
        else:
            label = f"a photo of {food_name}, {cuisine_type} food dish"

        labels.append(label)
        keys.append(key)

    _FOOD_CATEGORY_LABELS = labels
    _FOOD_CATEGORY_KEYS   = keys
    print(f"[FoodDetector] Built {len(labels)} food-category labels "
          f"({sum(1 for k in keys if k.lower() in VISUAL_DESCRIPTORS)} with rich descriptors).")
    return _FOOD_CATEGORY_LABELS, _FOOD_CATEGORY_KEYS


# ═══════════════════════════════════════════════════════════════
#  MAIN FUNCTION 1 — detect_food_local
#  Uses CLIP to check: is this image food or not?
# ═══════════════════════════════════════════════════════════════

def detect_food_local(image_path):
    if not _load_clip():
        return {
            "is_food":       None,
            "confidence":    0.0,
            "top_label":     "",
            "method":        "error",
            "error":         _load_error or "CLIP model could not be loaded",
            "food_score":    0.0,
            "nonfood_score": 0.0,
        }

    try:
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.LANCZOS)

        inputs = _clip_processor(
            text=ALL_LABELS,
            images=image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = _clip_model(**inputs)
            logits  = outputs.logits_per_image
            probs   = logits.softmax(dim=1)[0]

        food_probs    = probs[:FOOD_COUNT]
        nonfood_probs = probs[FOOD_COUNT:]

        food_score    = food_probs.sum().item()
        nonfood_score = nonfood_probs.sum().item()

        best_idx  = probs.argmax().item()
        top_label = ALL_LABELS[best_idx]

        is_food    = food_score > nonfood_score and food_score >= FOOD_THRESHOLD
        confidence = food_score if is_food else nonfood_score

        print(f"[FoodDetector] food_score={food_score:.3f} nonfood_score={nonfood_score:.3f} "
              f"is_food={is_food}")

        return {
            "is_food":       is_food,
            "confidence":    round(confidence, 3),
            "top_label":     top_label,
            "method":        "clip",
            "error":         None,
            "food_score":    round(food_score, 3),
            "nonfood_score": round(nonfood_score, 3),
        }

    except Exception as e:
        return {
            "is_food":       None,
            "confidence":    0.0,
            "top_label":     "",
            "method":        "error",
            "error":         str(e),
            "food_score":    0.0,
            "nonfood_score": 0.0,
        }


# ═══════════════════════════════════════════════════════════════
#  MAIN FUNCTION 2 — identify_food_type
#  Uses CLIP zero-shot with rich visual descriptors
# ═══════════════════════════════════════════════════════════════

def identify_food_type(image_path):
    """
    Uses CLIP zero-shot classification to identify WHAT food is in the image.
    Compares image against every recipe in RECIPE_DATABASE using rich visual
    descriptors and returns the best matching database key + confidence.
    Returns None if confidence is too low → caller shows No Recipe Found.
    """
    if not _load_clip():
        print("[FoodDetector] identify_food_type: CLIP not loaded.")
        return None

    category_labels, category_keys = _get_food_category_labels()

    if not category_labels:
        print("[FoodDetector] identify_food_type: No category labels available.")
        return None

    try:
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.LANCZOS)

        CHUNK     = 100
        all_probs = []

        for start in range(0, len(category_labels), CHUNK):
            chunk_labels = category_labels[start:start + CHUNK]
            inputs = _clip_processor(
                text=chunk_labels,
                images=image,
                return_tensors="pt",
                padding=True
            )
            with torch.no_grad():
                outputs      = _clip_model(**inputs)
                chunk_logits = outputs.logits_per_image[0]
            all_probs.append(chunk_logits)

        full_logits = torch.cat(all_probs, dim=0)
        full_probs  = full_logits.softmax(dim=0)

        best_idx  = full_probs.argmax().item()
        best_key  = category_keys[best_idx]
        best_conf = full_probs[best_idx].item()

        # Log top-5 for debugging
        top5 = full_probs.topk(min(5, len(category_keys)))
        print("[FoodDetector] Top-5 matches:")
        for rank, (score, idx) in enumerate(zip(
            top5.values.tolist(), top5.indices.tolist()
        )):
            print(f"  {rank+1}. '{category_keys[idx]}' — {score:.3f}")

        from database.food_dataset import RECIPE_DATABASE
        best_recipe = RECIPE_DATABASE.get(best_key, {})
        food_name   = best_recipe.get("food_name", best_key)

        # Reject if confidence is too low
        if best_conf < 0.15:
            print(f"[FoodDetector] Confidence too low ({best_conf:.3f}). Returning None.")
            return None

        print(f"[FoodDetector] Identified: '{food_name}' "
              f"(key='{best_key}', conf={best_conf:.3f})")

        return {
            "db_key":     best_key,
            "food_name":  food_name,
            "confidence": round(best_conf, 3),
            "method":     "clip"
        }

    except Exception as e:
        print(f"[FoodDetector] identify_food_type error: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  FILENAME CHECK — fast pre-filter before any AI runs
# ═══════════════════════════════════════════════════════════════

NON_FOOD_FILENAME_WORDS = {
    "selfie","portrait","person","people","man","woman","boy","girl",
    "baby","child","face","human","model","dog","cat","bird","animal",
    "pet","lion","tiger","horse","cow","snake","car","vehicle","truck",
    "bike","motorcycle","bus","train","plane","boat","ship","building",
    "house","office","school","hospital","hotel","church","mosque",
    "mountain","forest","beach","ocean","river","lake","sky","cloud",
    "sunset","flower","tree","plant","grass","garden","park","nature",
    "phone","laptop","computer","keyboard","screen","monitor","tablet",
    "book","document","paper","letter","note","poster","magazine",
    "chair","sofa","bed","furniture","room","clothes","shirt","dress",
    "shoes","bag","wallet","money","coin","flag","logo","art","painting",
    "drawing","cartoon","screenshot","diagram","chart","graph","pen",
    "pencil","stationery","toy","game","sport","tool","machine","road",
    "street","map","banner","sign",
}

FOOD_FILENAME_WORDS = {
    "food","meal","dish","recipe","cook","bake","grill","fry","roast",
    "lunch","dinner","breakfast","brunch","snack","dessert","drink",
    "plate","bowl","chicken","beef","pork","lamb","fish","shrimp",
    "rice","pasta","bread","noodle","ramen","spaghetti","salad","soup",
    "stew","curry","sauce","cheese","butter","cream","milk","egg",
    "pizza","burger","sandwich","taco","sushi","biryani","dosa","idli",
    "cake","cookie","brownie","pie","muffin","waffle","pancake","coffee",
    "tea","juice","smoothie","apple","banana","mango","orange","lemon",
    "tomato","potato","onion","garlic","pepper","carrot","broccoli",
    "chocolate","honey","sugar","spice","kebab","tikka","samosa","pav",
    "paratha","halwa","kheer","jalebi","dhokla","vada","chaat","pulao",
    "tandoori","masala","korma","vindaloo","dal","roti","chapati","naan",
    "gelato","tiramisu","croissant","baguette","crepe","quiche","tart",
    "sashimi","tempura","gyoza","onigiri","mochi","bento","pho","banh",
    "kimchi","bibimbap","tteok","bulgogi","pajeon","jjigae",
    "holige","puranpoli","puran","aloo","aloopara","stuffed",
    "rasgulla","rasmalai","bhelpuri","bhel","gulabjamun",
    "mutton","lamb","prawn","paneer","chole","rajma","haleem",
}

def check_filename(filename):
    """
    Returns: True (likely food), False (likely not food), None (uncertain)
    """
    name = filename.lower().replace("_", " ").replace("-", " ").rsplit(".", 1)[0]
    for w in NON_FOOD_FILENAME_WORDS:
        if w in name:
            return False
    for w in FOOD_FILENAME_WORDS:
        if w in name:
            return True
    return None


# ═══════════════════════════════════════════════════════════════
#  REJECTION DESCRIPTION — friendly message for non-food images
# ═══════════════════════════════════════════════════════════════

def describe_rejection(result):
    label      = result.get("top_label", "").lower()
    confidence = result.get("confidence", 0)
    pct        = int(confidence * 100)

    if "person" in label or "face" in label or "human" in label:
        return f"This looks like a photo of a person ({pct}% confident)."
    if "animal" in label or "pet" in label or "dog" in label or "cat" in label:
        return f"This looks like a photo of an animal ({pct}% confident)."
    if "car" in label or "vehicle" in label:
        return f"This looks like a photo of a vehicle ({pct}% confident)."
    if "nature" in label or "landscape" in label or "mountain" in label or "beach" in label:
        return f"This looks like a nature or landscape photo ({pct}% confident)."
    if "building" in label or "house" in label:
        return f"This looks like a photo of a building ({pct}% confident)."
    if "book" in label or "document" in label or "paper" in label:
        return f"This looks like a photo of a document or book ({pct}% confident)."
    if "phone" in label or "laptop" in label or "computer" in label:
        return f"This looks like a photo of an electronic device ({pct}% confident)."
    if "flower" in label or "tree" in label or "plant" in label:
        return f"This looks like a photo of plants or nature ({pct}% confident)."
    return f"This does not appear to be a food image ({pct}% confident it's not food)."