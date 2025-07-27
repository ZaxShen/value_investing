# Test file for Black formatting
def badly_formatted_function(x, y, z):
    result = x + y + z
    if result > 10:
        print("Result is greater than 10")
    else:
        print("Result is 10 or less")
    return result


# This should get formatted by Black when you save or run Format Document
data = {"key1": "value1", "key2": "value2", "key3": "value3"}
