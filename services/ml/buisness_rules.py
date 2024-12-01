def get_model_by_buisness_rules(model_dict, context, is_multi_model_enabled):
    if is_multi_model_enabled:
        if context == "Главная":
            return model_dict["decision_tree"]
        elif context == "Важно":
            return model_dict["kmeans"]
        elif context == "Способ подписания":
            return model_dict["decision_tree"]
    return model_dict["catboost"]

def get_basic_recommendation_text(context, prediction):
    if context =="главная страница":
        if prediction == "PayControl":
            text = "Воспользуйтесь быстрым и безопасным способом подписания из любой точки мира"
        elif prediction == "КЭП в приложении":
            text = "Подписывайте важные документы самым безопасным способом"
        elif prediction == "КЭП на токене":
            text = "Подписывайте важные документы даже со смартфона"
    elif context =="Бухгалтерия":
        if prediction == "PayControl":
            text = "Воспользуйтесь быстрым и безопасным способом подписания из любой точки мира"
        elif prediction == "КЭП в приложении":
            text = "Подписывайте важные документы самым безопасным способом"
        elif prediction == "КЭП на токене":
            text = "Подписывайте важные документы даже со смартфона"
    elif context =="платежи":
        if prediction == "PayControl":
            text = "Воспользуйтесь быстрым и безопасным способом подписания из любой точки мира"
        elif prediction == "КЭП в приложении":
            text = "Подписывайте важные документы самым безопасным способом"
        elif prediction == "КЭП на токене":
            text = "Подписывайте важные документы даже со смартфона"
    elif context =="выбор подписания":
        if prediction == "PayControl":
            text = "Воспользуйтесь быстрым и безопасным способом подписания из любой точки мира"
        elif prediction == "КЭП в приложении":
            text = "Подписывайте важные документы самым безопасным способом"
        elif prediction == "КЭП на токене":
            text = "Подписывайте важные документы даже со смартфона"
    elif context =="код подтверждения":
        if prediction == "PayControl":
            text = "Воспользуйтесь быстрым и безопасным способом подписания из любой точки мира"
        elif prediction == "КЭП в приложении":
            text = "Подписывайте важные документы самым безопасным способом"
        elif prediction == "КЭП на токене":
            text = "Подписывайте важные документы даже со смартфона"
    else:
        text = "Используйте PayControl - самый быстрый способ подписания"
    return text