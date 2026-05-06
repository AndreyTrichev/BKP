import os


EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
CE_MAX_LENGTH = int(os.environ.get("CE_MAX_LENGTH", "2048"))


QDRANT_URL = os.environ.get("QDRANT_URL", ":memory:")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "citizen_complaints")


CE_TEXT_MAX_CHARS = max(200, int(os.environ.get("CE_TEXT_MAX_CHARS", "900")))
CE_QUERY_MAX_CHARS = max(120, int(os.environ.get("CE_QUERY_MAX_CHARS", "500")))


BEAM_L2_LIMIT = max(1, int(os.environ.get("BEAM_L2_LIMIT", "40")))
BEAM_MIN = max(1, int(os.environ.get("BEAM_MIN", "2")))
W_L1 = float(os.environ.get("W_L1", "0.3"))
W_L2 = float(os.environ.get("W_L2", "0.7"))
REL_PRUNE = float(os.environ.get("REL_PRUNE", "0.8"))
CE_FINAL_LIMIT = max(1, int(os.environ.get("CE_FINAL_LIMIT", "10")))


SCORE_HIGH = float(os.environ.get("SCORE_HIGH", "0.347"))                                
GAP_HIGH = float(os.environ.get("GAP_HIGH", "0.004"))                                          
GAP_MIN = float(os.environ.get("GAP_MIN", "0.001"))                                    
RAW_FLOOR = float(os.environ.get("RAW_FLOOR", "-999.0"))                                           
                                                                                                          
                                                                                                          
MAXP_PER_DEPT = os.environ.get("MAXP_PER_DEPT", "on").lower() != "off"


DEPT_NAMES = {
    "urban_economy": "Хозяйство (УГХ)",
    "urban_development": "Городское развитие (УГР)",
    "education": "Образование",
    "culture": "Культура",
}

SUBDEPT_NAMES = {
             
    "gas": "Газоснабжение",
    "water": "Водоснабжение",
    "heat": "Теплоснабжение",
    "housing": "Содержание МКД",
    "roads": "Дороги и тротуары",
    "waste": "Вывоз мусора (ТКО)",
    "ecology": "Экология",
             
    "construction": "Строительство",
    "land": "Земельные участки",
    "planning": "Генплан и НТО",
    "trade": "Торговля",
    "beaches": "Пляжи и курорты",
    "tourism": "Туризм",
                     
    "preschool": "Детские сады",
    "school": "Школы",
    "custody": "Опека",
                  
    "institutions": "Учреждения культуры",
    "heritage": "Культурное наследие",
}
