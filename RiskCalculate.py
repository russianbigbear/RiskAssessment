import numpy as np


def read_data(filename):
    """Чтение данных"""
    with open(filename + ".dat") as file:
        size_matrix_str = int(file.readline())  # размер матрицы
        matrix = [[float(i) for i in file.readline().split()] for _ in range(size_matrix_str)]  # чтение матрицы

    return matrix


def print_data(matrix):
    """Вывод данных"""
    for i in range(len(matrix)):
        print("\t".join([str(round(k, 10)) for k in matrix[i]]))
    print("")
    
def create_riskmatrix(matrix):
    """Вычисление матрицы рисков"""

    max_in_columns = np.amax(matrix,axis=0)  # максимальные по столбцам
    risk_matrix = np.zeros((len(matrix),len(matrix[0])))  # матрица рисков

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            risk_matrix[i][j] = max_in_columns[j] - matrix[i][j]

    return risk_matrix


def criterion_vald(matrix):
    """Критерий Вальда(MaxMin, ищем наибольшую выгоду в худшем случае)"""

    min_str = [0 for i in range(len(matrix))]
    min_ind = [0 for i in range(len(matrix))]

    for i in range(len(matrix)):
        # ищем минимальное среди строк, запоминаем индекс
        min_str[i] = matrix[i][0]
        min_ind[i] = 0

        for j in range(len(matrix[0])):
            if min_str[i] > matrix[i][j]:
                min_ind[i] = j
                min_str[i] = matrix[i][j]

    tmp_ind = 0
    for i in range(len(min_str)):
        # ищем максимум среди минимув строк
        if min_str[tmp_ind] < min_str[i]:
            tmp_ind = i

    print("Оптимальная стратегия по Вальду: %d стратегия. Max = %f" % (tmp_ind + 1, min_str[tmp_ind]))


def criterion_savage(matrix):
    """Критерий Сэвиджа(MinMax, ищем меньший риск в худшем случае)"""

    max_str = [0 for i in range(len(matrix))]
    max_ind = [0 for i in range(len(matrix))]

    for i in range(len(matrix)):
        # ищем максимум среди строк, запоминаем индекс
        max_str[i] = matrix[i][0]
        max_ind[i] = 0

        for j in range(len(matrix[0])):
            if max_str[i] < matrix[i][j]:
                max_ind[i] = j
                max_str[i] = matrix[i][j]

    tmp_ind = 0
    for i in range(len(max_str)):
        # ищем минимумы среди максимумов строк
        if max_str[tmp_ind] > max_str[i]:
            tmp_ind = i

    print("Оптимальная стратегия по Севиджу: %d стратегия. Min = %f" % (tmp_ind + 1, max_str[tmp_ind]))


def criterion_hurwitz_cost(matrix, coof):
    """Критерий Гурвица(Средняя позиция выбора стратегии)(по платежной матрице)"""

    min_str = [0 for i in range(len(matrix))]
    min_ind = [0 for i in range(len(matrix))]
    max_str = [0 for i in range(len(matrix))]
    max_ind = [0 for i in range(len(matrix))]
    hurwitz_matrix = [0 for i in range(len(matrix))]

    for i in range(len(matrix)):
        # ищем максимум и минимум среди строк, запоминаем индекс
        min_str[i] = matrix[i][0]
        max_str[i] = matrix[i][0]
        min_ind[i] = 0
        max_ind[i] = 0

        for j in range(len(matrix[0])):
            if min_str[i] > matrix[i][j]:
                min_ind[i] = j
                min_str[i] = matrix[i][j]
            if max_str[i] < matrix[i][j]:
                max_ind[i] = j
                max_str[i] = matrix[i][j]

        hurwitz_matrix[i] = coof * min_str[i] + (1 - coof) * max_str[i] # считаем выражение по формуле

    tmp_ind = 0
    for i in range(len(matrix)):
        # ищем максимум в выражении
        if hurwitz_matrix[tmp_ind] < hurwitz_matrix[i]:
            tmp_ind = i

    print("Оптимальная стратегия по Гурвицу с коофицентом k = %f: %d стратегия. Max = %f (Платежная матрица)"
          % (coof, tmp_ind + 1, hurwitz_matrix[tmp_ind]))


def criterion_hurwitz_risk(matrix, coof):
    """Критерий Гурвица(Средняя позиция выбора стратегии)(по матрице рисков)"""

    min_str = [0 for i in range(len(matrix))]
    min_ind = [0 for i in range(len(matrix))]
    max_str = [0 for i in range(len(matrix))]
    max_ind = [0 for i in range(len(matrix))]
    hurwitz_matrix = [0 for i in range(len(matrix))]

    for i in range(len(matrix)):
        # ищем максимум и минимум среди строк, запоминаем индекс
        min_str[i] = matrix[i][0]
        max_str[i] = matrix[i][0]
        min_ind[i] = 0
        max_ind[i] = 0

        for j in range(len(matrix[0])):
            if min_str[i] > matrix[i][j]:
                min_ind[i] = j
                min_str[i] = matrix[i][j]
            if max_str[i] < matrix[i][j]:
                max_ind[i] = j
                max_str[i] = matrix[i][j]

        hurwitz_matrix[i] = coof * max_str[i] + (1 - coof) * min_str[i] # считаем выражение по формуле

    tmp_ind = 0
    for i in range(len(matrix)):
        # ищем минимум в выражении
        if hurwitz_matrix[tmp_ind] > hurwitz_matrix[i]:
            tmp_ind = i

    print("Оптимальная стратегия по Гурвицу с коофицентом k = %f: %d стратегия. Min = %f (Матрица рисков)"
          % (coof, tmp_ind + 1, hurwitz_matrix[tmp_ind]))


def perfect_experiment(matrix, probability):
    """Критерий, основанный на известных вероятностях условий"""

    risk_medium_matrix = [0 for i in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            risk_medium_matrix[i] += matrix[i][j] * probability[i]

    tmp_ind =0
    for i in range(len(matrix)):
        # ищем минимум в выражении
        if risk_medium_matrix[tmp_ind] > risk_medium_matrix[i]:
            tmp_ind = i

    print("Минимальное значение средних рисков равно %f."
          "Планированиеыше эксперимента выше этой цены становится нецелесообразным."% (risk_medium_matrix[tmp_ind]))

def main():
    """Основная функция"""

    cost_matrix = read_data(input("Введите имя файла: "))
    coof = float(input("Введите коофицент Гурвица: "))

    probability = [0 for i in range(len(cost_matrix[0]))]
    for i in range(len(cost_matrix[0])):
        probability[i] = float(input("Введите вероятность условия для столбца %d: " % (i + 1)))
    print()

    print("Платежная матрица: ")
    print_data(cost_matrix)
    print()

    risk_matrix = create_riskmatrix(cost_matrix)

    print("Матрица рисков: ")
    print_data(risk_matrix)
    print()

    criterion_vald(cost_matrix)
    criterion_savage(risk_matrix)
    criterion_hurwitz_cost(cost_matrix, coof)
    criterion_hurwitz_risk(risk_matrix, coof)
    perfect_experiment(risk_matrix, probability)


if __name__ == '__main__':
    main()