import numpy as np


def convolutional_layer(input_array, filters):
    # Получаем размеры входного массива
    input_depth, input_height, input_width = input_array.shape
    num_filters, _, filter_size, _ = filters.shape

    # Рассчитываем размер выходного массива
    output_height = input_height - filter_size + 1
    output_width = input_width - filter_size + 1

    # Преобразуем входной массив в im2col формат
    im2col_input = np.zeros((input_depth * filter_size * filter_size, output_height * output_width))
    for i in range(output_height):
        for j in range(output_width):
            patch = input_array[:, i:i + filter_size, j:j + filter_size].reshape(-1)
            im2col_input[:, i * output_width + j] = patch

    # Применяем фильтры к im2col формату
    im2col_output = np.dot(filters.reshape(num_filters, -1), im2col_input)

    # Преобразуем обратно в формат выходного массива
    output_array = im2col_output.reshape(num_filters, output_height, output_width)

    return output_array


# Прямая реализация свертки
def conv_direct(input_array, filters):
    input_depth, input_height, input_width = input_array.shape
    num_filters, _, filter_size, _ = filters.shape
    output_height = input_height - filter_size + 1
    output_width = input_width - filter_size + 1
    output_array = np.zeros((num_filters, output_height, output_width))

    for f in range(num_filters):
        for i in range(output_height):
            for j in range(output_width):
                output_array[f, i, j] = np.sum(input_array[:, i:i + filter_size, j:j + filter_size] * filters[f])

    return output_array


def main():
    # Генерируем тестовые данные
    input_array = np.random.rand(3, 5, 5)  # пример входного массива
    filter_size = 3  # размер фильтра
    num_filters = 2  # количество фильтров
    filters = np.random.rand(num_filters, 3, filter_size, filter_size)  # пример фильтра

    # Проверяем результаты
    output_im2col = convolutional_layer(input_array, filters)
    output_direct = conv_direct(input_array, filters)

    # Проверяем, совпадают ли результаты
    print(np.allclose(output_im2col, output_direct))


if __name__ == "__main__":
    main()
