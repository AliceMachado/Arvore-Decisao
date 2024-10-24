# Arvore-Decisao

Esta atividade tem como objetivo a implementação de um algoritmo de Árvore de Decisão, baseado na análise do desafio 'Iris Dataset', encontrado na plataforma Kaggle, onde é necessário avaliar o desempenho e ajustar os parâmetros para observarmos os impactos que podem ocorrer nas previsõers.

- Carregamento e preparação dos dados

#Importar bibliotecas para uso do Iris Dataset e as métricas
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Carregar o Iris Dataset
iris = load_iris()

#Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

- Implementação dos Modelos
	> Árvore de Decisão
#Importar as bibliotecas para a árvore de decisão
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

#Inicializando listas para armazenar os resultados
depths = [2, 3, 5, 10]
results = []

#Testar diferentes valores no max_depth
for depth in depths:
    model_gini = DecisionTreeClassifier(criterion='gini', max_depth=depth)
    model_gini.fit(X_train, y_train)
    y_pred_gini = model_gini.predict(X_test)
    accuracy_gini = accuracy_score(y_test, y_pred_gini)
    print("Profundidade da Árvore gini:", depth)
    print("Acurácia gini:", accuracy_score(y_test, y_pred_gini))
    print("Matriz de Confusão gini:\n", confusion_matrix(y_test, y_pred_gini))
    print("Relatório de Classificação gini:\n", classification_report(y_test, y_pred_gini))
    results.append((f'gini, max_depth={depth}', accuracy_gini))
    print("-------------------------------------------------------------")
!Imagem de exemplo dos resultados possíveis!
![diferentes-max_depth-criterio-gini](https://github.com/user-attachments/assets/5d168672-49f6-4994-9b0f-4c531b016bb1)

#Testar com critério 'entropy'
for depth in depths:
    model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    model_entropy.fit(X_train, y_train)
    y_pred_entropy = model_entropy.predict(X_test)
    accuracy_entropy = accuracy_score(y_test, y_pred_entropy)
    print("Profundidade da Árvore entropy:", depth)
    print("Acurácia entropy:", accuracy_score(y_test, y_pred_entropy))
    print("Matriz de Confusão entropy:\n", confusion_matrix(y_test, y_pred_entropy))
    print("Relatório de Classificação entropy:\n", classification_report(y_test, y_pred_entropy))
    results.append((f'gini, max_depth={depth}', accuracy_entropy))
    print("-------------------------------------------------------------")
!Imagem de exemplo dos resultados possíveis!
![diferentes-max_depth-criterio-entropy](https://github.com/user-attachments/assets/b8560e1c-3167-4a52-b294-d236eadcdcd9)

#Visualizar a árvore gerada para a melhor configuração
best_depth = max(results, key=lambda x: x[1])[0].split('=')[1]
best_criterion = 'gini' if 'gini' in max(results, key=lambda x: x[1])[0] else 'entropy'
model_best = DecisionTreeClassifier(criterion=best_criterion, max_depth=int(best_depth))
model_best.fit(X_train, y_train)
plt.figure(figsize=(20,10))
tree.plot_tree(model_best, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
!Imagem da árvore gerada!
![Arvore_decisao_gerada](https://github.com/user-attachments/assets/f726b4c8-745d-4b76-851a-6c39a25cef7c)

  > Comparando com modelo k-NN
#Comparando árvore de decisão com o modelo k-NN.
#Importar o modelo de classificação k-NN
from sklearn.neighbors import KNeighborsClassifier

#Lista de valores de k para testar
neighbors_values = [2, 3, 5, 10]

#Inicializar uma lista para armazenar os resultados
knn_results = []

#Testar diferentes valores de k
for k in neighbors_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("Profundidade da Árvore k-NN:", k)
    print("Acurácia k-NN:", accuracy_score(y_test, y_pred_knn))
    print("Matriz de Confusão k-NN:\n", confusion_matrix(y_test, y_pred_knn))
    print("Relatório de Classificação k-NN:\n", classification_report(y_test, y_pred_knn))
    knn_results.append((f'n_neighbors={n}', accuracy_knn))
    print("-------------------------------------------------------------")
!Imagem da classificação utilizando o modelo k-NN!
![diferentes_valores_k_modelo_k-NN](https://github.com/user-attachments/assets/78380691-9acf-424a-b175-a92b3981075b)


