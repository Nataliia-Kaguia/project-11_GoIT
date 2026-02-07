У папці graphics знаходятся графіки втрат та точності на валідаційному відборі

У папці model знаходится сама модель та інформація про неї

Інформація про модель включаючи архітектуру та результати оцінки:

'metrics': {

    'accuracy': 0.9284676582497405, 
    'precision': 0.9474089276552078, 
    'recall': 0.9220973782771535, 
    'f1': 0.9345818043780842, 
    'roc_auc': 0.9636173944710307}, 
'cv_auc_fast': 

    np.float64(0.9648659819935892), 
    'n_iter': 29,
    'architecture': (64, 32, 16)}

Функція активації: Relu

Оптимізатор: Adam

Більш детальная архітектура: {

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
}





