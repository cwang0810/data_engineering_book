# Volume II. Python Problem-Solving Workbook

A structured beginner-to-intermediate programming text with verified examples.

## Front Matter

This volume is a reproducible synthetic textbook artifact generated from verified seed tasks. Each chapter includes objectives, prerequisite tags, a worked example, and end-of-chapter checks.

## Table Of Contents

- Chapter 01 Function Design: Convert Degrees Radians
- Chapter 02 Function Design: Python Bell Number
- Chapter 03 Function Design: Lucid Number Smaller
- Chapter 04 Function Design: Check Integer Prime
- Chapter 05 Function Design: Python Minimum Possible
- Chapter 06 Function Design: Perfom Modulo Tuple
- Chapter 07 Function Design: Longest Chain Formed
- Chapter 08 Function Design: Average Value Numbers
- Chapter 09 Function Design: Check Number Jumps
- Chapter 10 Lists And Iteration: Python Remove Element
- Chapter 11 Lists And Iteration: Maximum Column Tuple
- Chapter 12 Lists And Iteration: Divide Lists Lambda
- Chapter 13 Lists And Iteration: Python Merge First
- Chapter 14 Lists And Iteration: Maximum Value Record
- Chapter 15 Lists And Iteration: Cumulative Values Present
- Chapter 16 String Algorithms: Python First Repeated
- Chapter 17 String Algorithms: Reverse Words String
- Chapter 18 String Algorithms: Search Literals String

## Main Text

## Chapter 01 Function Design: Convert Degrees Radians

Stage: Core
Topic: Function Design
Prerequisites: None

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Teach interface design before optimization so learners can read code contracts clearly.

Key terms
function contract, test case, helper variable, assertion

Lesson text
This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints. This section belongs to the topic `function_design`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to convert degrees to radians.
Programming exercise: Write a function to convert degrees to radians.
Reference implementation:
```python
import math
def radian_degree(degree):
 radian = degree*(math.pi/180)
 return radian
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 02 Function Design: Python Bell Number

Stage: Core
Topic: Function Design
Prerequisites: None

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Teach interface design before optimization so learners can read code contracts clearly.

Key terms
function contract, test case, helper variable, assertion

Lesson text
This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints. This section belongs to the topic `function_design`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a python function to find nth bell number.
Programming exercise: Write a python function to find nth bell number.
Reference implementation:
```python
def bell_Number(n): 
    bell = [[0 for i in range(n+1)] for j in range(n+1)] 
    bell[0][0] = 1
    for i in range(1, n+1):
        bell[i][0] = bell[i-1][i-1]
        for j in range(1, i+1): 
            bell[i][j] = bell[i-1][j-1] + bell[i][j-1] 
    return bell[n][0] 
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 03 Function Design: Lucid Number Smaller

Stage: Advanced
Topic: Function Design
Prerequisites: None

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Teach interface design before optimization so learners can read code contracts clearly.

Key terms
function contract, test case, helper variable, assertion

Lesson text
This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints. This section belongs to the topic `function_design`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to get a lucid number smaller than or equal to n.
Programming exercise: Write a function to get a lucid number smaller than or equal to n.
Reference implementation:
```python
def get_ludic(n):
	ludics = []
	for i in range(1, n + 1):
		ludics.append(i)
	index = 1
	while(index != len(ludics)):
		first_ludic = ludics[index]
		remove_index = index + first_ludic
		while(remove_index < len(ludics)):
			ludics.remove(ludics[remove_index])
			remove_index = remove_index + first_ludic - 1
		index += 1
	return ludics
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 04 Function Design: Check Integer Prime

Stage: Advanced
Topic: Function Design
Prerequisites: None

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Teach interface design before optimization so learners can read code contracts clearly.

Key terms
function contract, test case, helper variable, assertion

Lesson text
This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints. This section belongs to the topic `function_design`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to check if the given integer is a prime number.
Programming exercise: Write a function to check if the given integer is a prime number.
Reference implementation:
```python
def prime_num(num):
  if num >=1:
   for i in range(2, num//2):
     if (num % i) == 0:
                return False
     else:
                return True
  else:
          return False
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 05 Function Design: Python Minimum Possible

Stage: Advanced
Topic: Function Design
Prerequisites: None

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Teach interface design before optimization so learners can read code contracts clearly.

Key terms
function contract, test case, helper variable, assertion

Lesson text
This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints. This section belongs to the topic `function_design`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a python function to find minimum possible value for the given periodic function.
Programming exercise: Write a python function to find minimum possible value for the given periodic function.
Reference implementation:
```python
def floor_Min(A,B,N):
    x = max(B - 1,N)
    return (A*x) // B
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 06 Function Design: Perfom Modulo Tuple

Stage: Advanced
Topic: Function Design
Prerequisites: None

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Teach interface design before optimization so learners can read code contracts clearly.

Key terms
function contract, test case, helper variable, assertion

Lesson text
This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints. This section belongs to the topic `function_design`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to perfom the modulo of tuple elements in the given two tuples.
Programming exercise: Write a function to perfom the modulo of tuple elements in the given two tuples.
Reference implementation:
```python
def tuple_modulo(test_tup1, test_tup2):
  res = tuple(ele1 % ele2 for ele1, ele2 in zip(test_tup1, test_tup2)) 
  return (res) 
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 07 Function Design: Longest Chain Formed

Stage: Extension
Topic: Function Design
Prerequisites: None

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Teach interface design before optimization so learners can read code contracts clearly.

Key terms
function contract, test case, helper variable, assertion

Lesson text
This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints. This section belongs to the topic `function_design`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to find the longest chain which can be formed from the given set of pairs.
Programming exercise: Write a function to find the longest chain which can be formed from the given set of pairs.
Reference implementation:
```python
class Pair(object): 
	def __init__(self, a, b): 
		self.a = a 
		self.b = b 
def max_chain_length(arr, n): 
	max = 0
	mcl = [1 for i in range(n)] 
	for i in range(1, n): 
		for j in range(0, i): 
			if (arr[i].a > arr[j].b and
				mcl[i] < mcl[j] + 1): 
				mcl[i] = mcl[j] + 1
	for i in range(n): 
		if (max < mcl[i]): 
			max = mcl[i] 
	return max
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 08 Function Design: Average Value Numbers

Stage: Extension
Topic: Function Design
Prerequisites: None

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Teach interface design before optimization so learners can read code contracts clearly.

Key terms
function contract, test case, helper variable, assertion

Lesson text
This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints. This section belongs to the topic `function_design`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to find average value of the numbers in a given tuple of tuples.
Programming exercise: Write a function to find average value of the numbers in a given tuple of tuples.
Reference implementation:
```python
def average_tuple(nums):
    result = [sum(x) / len(x) for x in zip(*nums)]
    return result
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 09 Function Design: Check Number Jumps

Stage: Extension
Topic: Function Design
Prerequisites: None

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Teach interface design before optimization so learners can read code contracts clearly.

Key terms
function contract, test case, helper variable, assertion

Lesson text
This chapter highlights how to choose function boundaries, name helper variables, and preserve clarity under constraints. This section belongs to the topic `function_design`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to check for the number of jumps required of given length to reach a point of form (d, 0) from origin in a 2d plane.
Programming exercise: Write a function to check for the number of jumps required of given length to reach a point of form (d, 0) from origin in a 2d plane.
Reference implementation:
```python
def min_Jumps(a, b, d): 
    temp = a 
    a = min(a, b) 
    b = max(temp, b) 
    if (d >= b): 
        return (d + b - 1) / b 
    if (d == 0): 
        return 0
    if (d == a): 
        return 1
    else:
        return 2
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 10 Lists And Iteration: Python Remove Element

Stage: Advanced
Topic: Lists And Iteration
Prerequisites: Function Design

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Develop traversal habits and accumulator patterns before introducing more compact idioms.

Key terms
loop, accumulator, transformation, condition

Lesson text
This chapter explains traversal patterns, accumulator variables, and how to transform list data into reliable outputs. This section belongs to the topic `lists_and_iteration`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a python function to remove the k'th element from a given list.
Programming exercise: Write a python function to remove the k'th element from a given list.
Reference implementation:
```python
def remove_kth_element(list1, L):
    return  list1[:L-1] + list1[L:]
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 11 Lists And Iteration: Maximum Column Tuple

Stage: Advanced
Topic: Lists And Iteration
Prerequisites: Function Design

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Develop traversal habits and accumulator patterns before introducing more compact idioms.

Key terms
loop, accumulator, transformation, condition

Lesson text
This chapter explains traversal patterns, accumulator variables, and how to transform list data into reliable outputs. This section belongs to the topic `lists_and_iteration`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to find the maximum of nth column from the given tuple list.
Programming exercise: Write a function to find the maximum of nth column from the given tuple list.
Reference implementation:
```python
def max_of_nth(test_list, N):
  res = max([sub[N] for sub in test_list])
  return (res) 
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 12 Lists And Iteration: Divide Lists Lambda

Stage: Advanced
Topic: Lists And Iteration
Prerequisites: Function Design

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Develop traversal habits and accumulator patterns before introducing more compact idioms.

Key terms
loop, accumulator, transformation, condition

Lesson text
This chapter explains traversal patterns, accumulator variables, and how to transform list data into reliable outputs. This section belongs to the topic `lists_and_iteration`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to divide two lists using map and lambda function.
Programming exercise: Write a function to divide two lists using map and lambda function.
Reference implementation:
```python
def div_list(nums1,nums2):
  result = map(lambda x, y: x / y, nums1, nums2)
  return list(result)
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 13 Lists And Iteration: Python Merge First

Stage: Extension
Topic: Lists And Iteration
Prerequisites: Function Design

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Develop traversal habits and accumulator patterns before introducing more compact idioms.

Key terms
loop, accumulator, transformation, condition

Lesson text
This chapter explains traversal patterns, accumulator variables, and how to transform list data into reliable outputs. This section belongs to the topic `lists_and_iteration`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a python function to merge the first and last elements separately in a list of lists.
Programming exercise: Write a python function to merge the first and last elements separately in a list of lists.
Reference implementation:
```python
def merge(lst):  
    return [list(ele) for ele in list(zip(*lst))] 
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 14 Lists And Iteration: Maximum Value Record

Stage: Extension
Topic: Lists And Iteration
Prerequisites: Function Design

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Develop traversal habits and accumulator patterns before introducing more compact idioms.

Key terms
loop, accumulator, transformation, condition

Lesson text
This chapter explains traversal patterns, accumulator variables, and how to transform list data into reliable outputs. This section belongs to the topic `lists_and_iteration`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to find the maximum value in record list as tuple attribute in the given tuple list.
Programming exercise: Write a function to find the maximum value in record list as tuple attribute in the given tuple list.
Reference implementation:
```python
def maximum_value(test_list):
  res = [(key, max(lst)) for key, lst in test_list]
  return (res) 
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 15 Lists And Iteration: Cumulative Values Present

Stage: Extension
Topic: Lists And Iteration
Prerequisites: Function Design

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Develop traversal habits and accumulator patterns before introducing more compact idioms.

Key terms
loop, accumulator, transformation, condition

Lesson text
This chapter explains traversal patterns, accumulator variables, and how to transform list data into reliable outputs. This section belongs to the topic `lists_and_iteration`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to find the cumulative sum of all the values that are present in the given tuple list.
Programming exercise: Write a function to find the cumulative sum of all the values that are present in the given tuple list.
Reference implementation:
```python
def cummulative_sum(test_list):
  res = sum(map(sum, test_list))
  return (res)
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 16 String Algorithms: Python First Repeated

Stage: Advanced
Topic: String Algorithms
Prerequisites: Lists And Iteration

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Highlight local state updates, boundary checks, and repeated scans.

Key terms
scan, index, character test, state update

Lesson text
This chapter introduces string scanning patterns, character bookkeeping, and small helper conditions. This section belongs to the topic `string_algorithms`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a python function to find the first repeated character in a given string.
Programming exercise: Write a python function to find the first repeated character in a given string.
Reference implementation:
```python
def first_repeated_char(str1):
  for index,c in enumerate(str1):
    if str1[:index+1].count(c) > 1:
      return c 
  return "None"
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 17 String Algorithms: Reverse Words String

Stage: Advanced
Topic: String Algorithms
Prerequisites: Lists And Iteration

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Highlight local state updates, boundary checks, and repeated scans.

Key terms
scan, index, character test, state update

Lesson text
This chapter introduces string scanning patterns, character bookkeeping, and small helper conditions. This section belongs to the topic `string_algorithms`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to reverse words in a given string.
Programming exercise: Write a function to reverse words in a given string.
Reference implementation:
```python
def reverse_words(s):
        return ' '.join(reversed(s.split()))
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.


## Chapter 18 String Algorithms: Search Literals String

Stage: Extension
Topic: String Algorithms
Prerequisites: Lists And Iteration

Learning objectives
- Read a function prompt and identify the required input-output contract.
- Use tests to confirm correctness on normal and edge cases.
- Explain why the chosen control flow matches the problem structure.

Why this chapter matters
Highlight local state updates, boundary checks, and repeated scans.

Key terms
scan, index, character test, state update

Lesson text
This chapter introduces string scanning patterns, character bookkeeping, and small helper conditions. This section belongs to the topic `string_algorithms`. Learners should identify the data transformation, choose a function signature, and confirm correctness with assertions.

Worked example
Question: Write a function to search a literals string in a string and also find the location within the original string where the pattern occurs by using regex.
Programming exercise: Write a function to search a literals string in a string and also find the location within the original string where the pattern occurs by using regex.
Reference implementation:
```python
import re
pattern = 'fox'
text = 'The quick brown fox jumps over the lazy dog.'
def find_literals(text, pattern):
  match = re.search(pattern, text)
  s = match.start()
  e = match.end()
  return (match.re.pattern, s, e)
```
Verification coverage: 3 assertion(s).

Common mistakes
- Returning too early before all input elements are processed.
- Ignoring an edge case covered by the provided assertions.
- Writing code that works on one example but not on the full test set.

End-of-chapter checks
- State the main idea of the chapter in one sentence.
- Re-solve the checkpoint exercise without looking at the example.
- Explain one mistake that would lead to a wrong answer.

