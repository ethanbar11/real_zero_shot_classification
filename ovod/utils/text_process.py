
def is_subsequence(s, t):
    """Check if s is a subsequence of t."""
    it = iter(t)
    return all(char in it for char in s)

def check_percent(A, B, percent=90):
    """Check if A is a subsequence of B, where A is a string and B is a string or a list of strings.
    If B is a list of strings, check if A is a subsequence of any of the strings in B.
    If A is a subsequence of B, return True. Otherwise, return False.
    """
    len_percent = int(0.01 * percent * len(A))
    for i in range(len(A) - len_percent + 1):
        if is_subsequence(A[i:i + len_percent], B):
            return True
    return False



# main
if __name__ == '__main__':
    A = "hello"
    B = "ahsepllyo"
    print(check_percent(A, B))  # This will print True since "hell" (90% of "hello") is a subsequence in "ahsepllyo".