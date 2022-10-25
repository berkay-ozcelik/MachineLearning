def calculate_gradient(w_0: float, w_1: float, x: [float], y: [float]) -> (float, float):
    n = len(x)
    respect_to_w0 = 0
    respect_to_w1 = 0

    for i in range(n):
        respect_to_w0 += (w_0 + w_1 * x[i] - y[i])
        respect_to_w1 += (w_0 * x[i] + w_1 * (x[i] ** 2) - x[i] * y[i])
    respect_to_w0 *= 2
    respect_to_w1 *= 2
    return respect_to_w0, respect_to_w1


def linear_regression(x: [float], y: [float]) -> (float, float):
    w_0 = 0.5
    w_1 = 0.5
    threshold = 0.00000001
    learning_rate = 0.00001
    d_w0, d_w1 = calculate_gradient(w_0, w_1, x, y)
    while abs(d_w0) > threshold and abs(d_w1) > threshold:
        if not abs(d_w0) < threshold:
            w_0 -= d_w0 * learning_rate
        if not abs(d_w1) < threshold:
            w_1 -= d_w1 * learning_rate
        d_w0, d_w1 = calculate_gradient(w_0, w_1, x, y)

    return w_0, w_1


def main():
    x = [i for i in range(10)]
    y = [i * 2 for i in range(10)]

    w_0, w_1 = linear_regression(x, y)
    print("w_0 is ", w_0, " and w_1 is ", w_1)


if __name__ == "__main__":
    main()
