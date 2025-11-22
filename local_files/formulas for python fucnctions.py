leap year formula⬇️
if (year %4 == 0 and year %100!=0) or (year%400==0):


# recurtion in python
def factorial(num):
    if num == 1 or num == 0:  # Base case: factorial of 0 or 1 is 1
        return 1
    else:
        return num * factorial(num - 1)  # Recursive call

# Input from the user to calculate factorial
num = int(input('Enter your number: '))
boom = factorial(num)  # Call the factorial function
print(boom)  # Print the result

# Function to calculate the nth Fibonacci number using recursion
def fibonacchi(n):
    if n == 0 or n == 1:  # Base cases: Fibonacci of 0 or 1 is 1
        return 1
    else:
        return fibonacchi(n - 1) + fibonacchi(n - 2)  # Recursive call

# Input from the user to calculate Fibonacci number
n = int(input('Enter your number: '))
xyz = fibonacchi(n)  # Call the Fibonacci function
print(xyz)  # Print the result




import math
def cockroach_speed(s):
    """
    Converts speed from kilometers per hour (km/h) to centimeters per second (cm/s) 
    and returns the largest integer less than or equal to the result.

    Parameters:
    s (float): The speed in kilometers per hour (km/h).

    Returns:
    int: The speed converted to centimeters per second (cm/s), rounded down to the nearest integer.

    Notes:
    - The conversion factor from km/h to cm/s is 100000/3600.
    - The `math.floor` function is used to round down the result to the nearest integer.
    """
    # Good Luck!
    sec_cm_h=s*(100000/3600)
    return math.floor(sec_cm_h)


name='sachin'
print(name)

# Number of letters in the English alphabet
MAX_CHAR = 26

# Function to check if a string is a pangram
def is_pangram(s):
    # Create a list to track which letters have been seen
    vis = [False] * MAX_CHAR
    for c in s:
        # Mark uppercase letters as seen
        if 'A' <= c <= 'Z':
            vis[ord(c) - ord('A')] = True
        # Mark lowercase letters as seen
        elif 'a' <= c <= 'z':
            vis[ord(c) - ord('a')] = True
    # If any letter is missing, return False
    for flag in vis:
        if not flag:
            return False
    # All letters found, return True
    return True

# Example usage
s = "The quick brown fox jumps over the dog"
print("true" if is_pangram(s) else "false")
