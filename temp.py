def iq_test(numbers):
    # your code here
    arr = [int(x) % 2 for x in numbers.split()]
    print(arr)
    p, q, r = 0, 1, 2
    while r < len(numbers):
        if arr[p] % 2 != arr[q] % 2 and arr[p] % 2 != arr[r] % 2:
            return p
        if arr[q] % 2 != arr[p] % 2 and arr[q] % 2 != arr[r] % 2:
            return q
        if arr[r] % 2 != arr[p] % 2 and arr[r] % 2 != arr[q] % 2:
            return r
        p += 1
        q += 1
        r += 1


if __name__ == "__main__":
    print(iq_test("2 4 7 8 10"))
