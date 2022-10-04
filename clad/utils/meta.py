SODA_DOMAINS = {
    "period": ["Daytime", "Night"],
    "weather": ["Clear", "Overcast", "Rainy"],
    "city": ["Guangzhou", "Shenzhen", "Shanghai"],
    "location": ["Citystreet", "Countryroad", "Highway"]
}

CLADD_TRAIN_VAL_DOMAINS = [
    {'city': 'Shanghai', 'location': 'Citystreet', 'period': 'Daytime', 'weather': 'Clear'},
    {'location': 'Highway', 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
    {'period': 'Night'},
    {'period': 'Daytime', 'weather': 'Rainy'}]

CLADD_TEST_DOMAINS = [
    {'location': ['Citystreet', 'Countryroad'], 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
    {'location': 'Highway', 'period': 'Daytime', 'weather': ['Clear', 'Overcast']},
    {'period': 'Night'},
    {'period': 'Daytime', 'weather': 'Rainy'}]

SODA_CATEGORIES = {
    1: "Pedestrain",
    2: "Cyclist",
    3: "Car",
    4: "Truck",
    5: "Tram (Bus)",
    6: "Tricycle"
}

START_NIGHT = "195035"
