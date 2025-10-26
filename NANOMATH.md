# Math Explained Super Simply

**A companion guide to NANO.md that explains every mathematical concept in the simplest possible way**

If you're not great at math, this guide is for you! Every time NANO.md uses a math concept, come here to understand it from the ground up.

---

## Table of Contents
1. [Basic Number Concepts](#basic-number-concepts)
2. [Vectors - Lists of Numbers](#vectors---lists-of-numbers)
3. [Operations on Numbers](#operations-on-numbers)
4. [Averages and Statistics](#averages-and-statistics)
5. [Percentages and Probabilities](#percentages-and-probabilities)
6. [Logarithms - The Opposite of Powers](#logarithms---the-opposite-of-powers)
7. [Gradients - How Things Change](#gradients---how-things-change)
8. [Matrix Operations](#matrix-operations)
9. [Trigonometry Basics](#trigonometry-basics)
10. [Special AI Functions](#special-ai-functions)

---

## Basic Number Concepts

###Positive and Negative Numbers

**Positive numbers** are numbers greater than zero: 1, 2, 3.5, 100

**Negative numbers** are numbers less than zero: -1, -2, -3.5, -100

```
Number line:
<───|───|───|───|───|───|───|───|───|───>
   -4  -3  -2  -1   0   1   2   3   4
    ↑                   ↑
 negative           positive
```

**Why this matters in AI:** AI models use both positive and negative numbers to represent different concepts.

### Integers vs. Decimals

**Integer:** A whole number (no decimal point)
- Examples: 1, 2, 42, -5, 0

**Decimal (Float):** A number with a decimal point
- Examples: 1.5, 3.14, -0.5, 2.0

**Why this matters in AI:** Token IDs are integers, but the model's internal calculations use decimals.

### Exponents (Powers)

**What it means:** Multiply a number by itself multiple times

```
2^1 = 2              (2 once)
2^2 = 2 × 2 = 4      (2 twice)
2^3 = 2 × 2 × 2 = 8  (2 three times)
2^16 = 65,536        (2 sixteen times - this is nanochat's vocab size!)
```

**Read as:** "2 to the power of 3" or "2 cubed" (for 3) or "2 squared" (for 2)

**Symbol:** `^` or written as superscript (2³)

**Why this matters in AI:** Vocabulary size is 2^16 = 65,536 tokens

---

## Vectors - Lists of Numbers

### What is a Vector?

**Simple explanation:** A vector is just a list of numbers in a specific order.

**Example:**
```
v = [0.2, -0.5, 0.8, 0.1]
```

This is a vector with 4 numbers (4 dimensions).

### Why Lists in Brackets?

The square brackets `[]` mean "these numbers belong together as one unit."

**Examples:**
```
[1, 2, 3]          ← A 3-dimensional vector
[0.5, -0.2]        ← A 2-dimensional vector
[1.1, 2.2, 3.3, 4.4, 5.5]  ← A 5-dimensional vector
```

### Dimensions

**Dimension** = how many numbers are in the list

```
[5]                  ← 1 dimension
[5, 10]              ← 2 dimensions
[5, 10, 15]          ← 3 dimensions
[5, 10, 15, 20]      ← 4 dimensions
```

**In nanochat:** The d20 model uses 1280-dimensional vectors!

```
[0.1, -0.2, 0.5, ..., 0.3, -0.1]
 ↑                           ↑
 first                    1280th
 number                   number
```

### Why Vectors?

**Simple answer:** Vectors let us represent complex concepts with numbers.

**Example:** Representing the word "cat"

Instead of just one number, we use many numbers to capture different aspects:
```
"cat" → [0.8,  0.9, 0.1, -0.2, 0.5, ...]
         ↑     ↑    ↑     ↑     ↑
      animal? pet? wild? size? cute?
```

Each dimension might represent some feature (though in practice, the model learns these automatically).

### Vector Notation

Different ways to write the same thing:

```
v = [1, 2, 3]        ← Programming style
v = (1, 2, 3)        ← Math textbook style
     ⎡1⎤
v =  ⎢2⎥             ← Column vector style
     ⎣3⎦
```

They all mean the same thing: a list of numbers.

---

## Operations on Numbers

### Addition and Subtraction

**Addition:** Combine two numbers

```
3 + 5 = 8
-2 + 7 = 5
0.5 + 0.3 = 0.8
```

**Subtraction:** Take one number away from another

```
10 - 3 = 7
5 - 8 = -3
0.8 - 0.2 = 0.6
```

### Multiplication

**What it means:** Add a number to itself multiple times

```
3 × 4 = 12        (3 + 3 + 3 + 3)
2 × 5 = 10        (2 + 2 + 2 + 2 + 2)
```

**Symbols:** `×`, `*`, or `·`

```
3 × 4 = 12
3 * 4 = 12        ← Used in programming
3 · 4 = 12        ← Used in math
```

**Special cases:**
```
Multiplying by 0:   5 × 0 = 0
Multiplying by 1:   5 × 1 = 5
Multiplying by -1:  5 × -1 = -5  (flips the sign)
```

### Division

**What it means:** Split a number into equal parts

```
12 ÷ 3 = 4       (How many 3s fit in 12?)
10 / 2 = 5       (Split 10 into 2 equal parts)
```

**Symbols:** `÷`, `/`, or written as a fraction

```
12 ÷ 3 = 4
12 / 3 = 4       ← Used in programming
12
-- = 4           ← Fraction style
3
```

**Special cases:**
```
Dividing by 1:   10 / 1 = 10
Dividing by itself:  10 / 10 = 1
Can't divide by 0:   10 / 0 = ERROR!
```

### Square Root (√)

**What it means:** What number, multiplied by itself, gives you this number?

```
√4 = 2          because  2 × 2 = 4
√9 = 3          because  3 × 3 = 9
√16 = 4         because  4 × 4 = 16
√25 = 5         because  5 × 5 = 25
```

**Symbol:** `√` or `sqrt()` in programming

**Why this matters in AI:** Used in normalization (RMSNorm) to keep numbers from getting too big.

### Squaring (x²)

**What it means:** Multiply a number by itself

```
2² = 2 × 2 = 4
3² = 3 × 3 = 9
5² = 5 × 5 = 25
(-2)² = (-2) × (-2) = 4   ← Negative × Negative = Positive!
```

**Read as:** "2 squared" or "2 to the power of 2"

**Why this matters in AI:** Used in ReLU² activation and in calculating variance.

### Dot Product (·)

**What it means:** Multiply corresponding numbers and add them up

**Given two vectors:**
```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = (1×4) + (2×5) + (3×6)
      = 4 + 10 + 18
      = 32
```

**Step by step:**
```
Position 1:  1 × 4 = 4
Position 2:  2 × 5 = 10
Position 3:  3 × 6 = 18
Add them:    4 + 10 + 18 = 32
```

**Why this matters in AI:** This is how Query and Key vectors are compared in attention!

**Visual example:**
```
Query:   [2,   3,   1]      "What am I looking for?"
  ×
Key:     [1,   2,   4]      "What do I have?"
  =
         2 +  6 +  4  = 12  ← Attention score!
```

Higher score = more attention!

---

## Averages and Statistics

### Mean (Average)

**What it means:** Add all numbers and divide by how many numbers there are

**Example:**
```
Numbers: 2, 4, 6

Step 1: Add them up
2 + 4 + 6 = 12

Step 2: Divide by count (we have 3 numbers)
12 ÷ 3 = 4

Mean = 4
```

**Formula:**
```
mean = (sum of all numbers) / (count of numbers)
```

**Symbol:** `μ` (Greek letter mu) or just "mean"

**Why this matters in AI:** Used in normalization to center the data.

### Sum (Σ)

**Symbol:** `Σ` (Greek letter sigma)

**What it means:** Add up all the numbers

**Example:**
```
Σ of [1, 2, 3, 4] = 1 + 2 + 3 + 4 = 10
```

**Read as:** "Sum of..." or "Sigma"

**Formal notation:**
```
  4
  Σ  i  =  1 + 2 + 3 + 4 = 10
 i=1

Read as: "Sum of i, from i=1 to 4"
```

### Variance and Standard Deviation

**Variance:** How spread out are the numbers?

**Steps to calculate:**
```
Numbers: 2, 4, 6

Step 1: Find the mean
mean = (2 + 4 + 6) / 3 = 4

Step 2: Subtract mean from each number
2 - 4 = -2
4 - 4 = 0
6 - 4 = 2

Step 3: Square each difference
(-2)² = 4
(0)² = 0
(2)² = 4

Step 4: Average these squared differences
variance = (4 + 0 + 4) / 3 = 2.67
```

**Standard Deviation:** Just the square root of variance

```
std = √variance = √2.67 = 1.63
```

**Why this matters in AI:** Used in normalization to scale the data.

---

## Percentages and Probabilities

### Percentages (%)

**What it means:** "Out of 100"

```
50% = 50 out of 100 = half
25% = 25 out of 100 = one quarter
100% = all of it
0% = none of it
```

**Converting between percentage and decimal:**
```
50% = 0.50 = 0.5
25% = 0.25
100% = 1.0
1% = 0.01
```

**Trick:** Move the decimal point two places

```
45% → 0.45  (move decimal left 2 places)
0.75 → 75%  (move decimal right 2 places)
```

### Probability

**What it means:** How likely is something to happen? (between 0 and 1)

```
Probability 0.0 (0%)   = impossible
Probability 0.5 (50%)  = maybe, maybe not
Probability 1.0 (100%) = certain
```

**Example: Coin flip**
```
Heads: probability = 0.5 (50%)
Tails: probability = 0.5 (50%)
Total: 0.5 + 0.5 = 1.0 (100%)  ← Probabilities always sum to 1
```

**In AI:**
```
Model's prediction for next token:
"cat"  → 0.45 (45%)
"dog"  → 0.30 (30%)
"bird" → 0.15 (15%)
other  → 0.10 (10%)
Total  →  1.00 (100%)
```

---

## Logarithms - The Opposite of Powers

### What is a Logarithm?

**Simple question it answers:** "What power do I raise this number to, to get that number?"

**Example:**
```
2³ = 8
So: log₂(8) = 3

Read as: "log base 2 of 8 equals 3"
Meaning: "2 to what power gives 8? Answer: 3"
```

### Common Logarithms

**Base 2 (log₂):** Used in computer science
```
log₂(2) = 1     because 2¹ = 2
log₂(4) = 2     because 2² = 4
log₂(8) = 3     because 2³ = 8
log₂(16) = 4    because 2⁴ = 16
```

**Base 10 (log₁₀):** Regular logarithm
```
log₁₀(10) = 1    because 10¹ = 10
log₁₀(100) = 2   because 10² = 100
log₁₀(1000) = 3  because 10³ = 1000
```

**Natural log (ln or logₑ):** Base e ≈ 2.718...
```
This is the one used in AI!
```

### Negative Logarithm (-log)

**Used in loss functions!**

**Important property:** `-log` of a small number = big number

```
-log(1.0) = 0      ← Perfect prediction! (100% confidence)
-log(0.5) ≈ 0.69   ← 50% confidence
-log(0.1) ≈ 2.30   ← 10% confidence (bad prediction)
-log(0.01) ≈ 4.61  ← 1% confidence (very bad!)
```

**Visual:**
```
Probability   -log(probability)
1.0 (100%) →  0.00   ← No loss (perfect!)
0.5 (50%)  →  0.69   ← Some loss
0.1 (10%)  →  2.30   ← High loss (bad!)
0.01 (1%)  →  4.61   ← Very high loss (very bad!)
```

**Why this matters in AI:** Loss = -log(probability of correct answer)

If model predicts correct token with:
- 100% confidence → loss = 0 (perfect!)
- 50% confidence → loss = 0.69
- 10% confidence → loss = 2.30 (needs improvement!)

### Log Rules (Simplified)

**Adding logs:**
```
log(a) + log(b) = log(a × b)
```

**Subtracting logs:**
```
log(a) - log(b) = log(a ÷ b)
```

**Multiplying log:**
```
k × log(a) = log(a^k)
```

---

## Gradients - How Things Change

### What is a Gradient?

**Simple explanation:** How much does the output change when we change the input a tiny bit?

**Analogy:** You're on a hill. The gradient tells you:
- Which direction is uphill
- Which direction is downhill
- How steep the slope is

### Rate of Change

**Example: Driving a car**

```
Time    Distance
0 min   0 miles
1 min   1 mile
2 min   2 miles
3 min   3 miles

Rate of change = 1 mile per minute
(For every 1 minute, distance increases by 1 mile)
```

**In math:**
```
change in distance      3 miles - 0 miles     3
─────────────────── = ───────────────────  = ─ = 1 mile/min
change in time          3 min - 0 min         3
```

### Derivative (∂ or d/dx)

**Symbol:** `∂` (partial derivative) or `d/dx` (derivative)

**Read as:** "partial derivative of" or "dee by dee ex"

**What it means:** Rate of change at a specific point

**Example:**
```
f(x) = x²

At x = 2:
f(2) = 4

Derivative: df/dx = 2x
At x = 2: df/dx = 2(2) = 4

Meaning: "At x=2, if we increase x by tiny bit, f increases 4 times as fast"
```

**Visual:**
```
      f(x) = x²
       ↑
    9  •
       │   ← Steep slope (gradient = 6)
    4  •
       │ ← Less steep (gradient = 4)
    1  • ← Even less (gradient = 2)
    0  •─────→ x
       0 1 2 3
```

### Gradient in AI

**In training:** Gradient tells us how to adjust weights to reduce loss

```
Loss = 5.0
Weight = 2.0

Gradient (∂Loss/∂Weight) = -3.0

This means:
- If we INCREASE weight by 0.1, loss DECREASES by 0.3
- If we DECREASE weight by 0.1, loss INCREASES by 0.3

So we should INCREASE the weight!
```

**Gradient Descent:**
```
new_weight = old_weight - learning_rate × gradient
           = 2.0 - 0.01 × (-3.0)
           = 2.0 + 0.03
           = 2.03
```

**Visual: Going downhill**
```
     High Loss
        ╱╲
       ╱  ╲        ← Start here
      ╱ ●  ╲           ↓
     ╱   ●  ╲       Follow gradient
    ╱     ●  ╲         ↓
   ╱_______●__╲    End here (low loss)
      Bottom
```

### Chain Rule

**Simple explanation:** If A affects B, and B affects C, then A affects C

**Example:**
```
temperature affects ice cream sales
ice cream sales affect shop revenue

So: temperature affects shop revenue (through ice cream sales)
```

**In AI:**
```
Weights affect activations
Activations affect output
Output affects loss

So: Weights affect loss (through activations and output)

This is how backpropagation works!
```

---

## Matrix Operations

### What is a Matrix?

**Simple explanation:** A matrix is a grid of numbers (like a spreadsheet)

**Example:**
```
     ⎡ 1  2  3 ⎤
M =  ⎢ 4  5  6 ⎥
     ⎣ 7  8  9 ⎦

This is a 3×3 matrix (3 rows, 3 columns)
```

**Size notation:** rows × columns

```
⎡ 1  2 ⎤
⎣ 3  4 ⎦  ← 2×2 matrix (2 rows, 2 columns)

⎡ 1  2  3 ⎤
⎣ 4  5  6 ⎦  ← 2×3 matrix (2 rows, 3 columns)
```

### Matrix-Vector Multiplication

**How it works:** Each row of the matrix does a dot product with the vector

**Example:**
```
Matrix × Vector:

⎡ 1  2 ⎤     ⎡ 5 ⎤     ⎡ (1×5 + 2×6) ⎤     ⎡ 17 ⎤
⎣ 3  4 ⎦  ×  ⎣ 6 ⎦  =  ⎣ (3×5 + 4×6) ⎦  =  ⎣ 39 ⎦

Row 1: (1×5) + (2×6) = 5 + 12 = 17
Row 2: (3×5) + (4×6) = 15 + 24 = 39
```

**Why this matters in AI:** Every layer in the neural network does this operation!

```
Input vector → Matrix (weights) → Output vector

[x₁, x₂] → ⎡w₁ w₂⎤ → [y₁, y₂]
            ⎣w₃ w₄⎦
```

### Transpose

**What it means:** Flip the matrix over its diagonal

**Example:**
```
Original:        Transposed:
⎡ 1  2  3 ⎤      ⎡ 1  4 ⎤
⎣ 4  5  6 ⎦      ⎢ 2  5 ⎥
                 ⎣ 3  6 ⎦

Rows become columns
Columns become rows
```

**Symbol:** `Aᵀ` or `A.T` in programming

**Why this matters in AI:** Used in backpropagation to reverse the flow of gradients.

---

## Trigonometry Basics

### Sine and Cosine

**What they are:** Functions that create wave patterns

**Simple visualization:**
```
Imagine a point moving in a circle:

        (0,1)
          •
          │
(-1,0)•───┼───•(1,0)
          │
          •
        (0,-1)

As it goes around:
- x-coordinate = cos(angle)
- y-coordinate = sin(angle)
```

**Values for common angles:**
```
Angle    cos    sin
0°       1.0    0.0
90°      0.0    1.0
180°    -1.0    0.0
270°     0.0   -1.0
360°     1.0    0.0  (back to start)
```

**Wave visualization:**
```
sin(x):
1 │  ╭─╮       ╭─╮
0 │──┼─┴──╮ ╭──┴─┼──→ x
-1│      ╰─╯

cos(x):
1 │─╮       ╭─╮
0 │ ┴──╮ ╭──┴
-1│    ╰─╯
```

**Why this matters in AI:** Rotary embeddings use sin and cos to encode position!

### Rotation with Sin and Cos

**How to rotate a point:**

```
Original point: (x, y)
Rotate by angle θ

New point:
x' = x × cos(θ) - y × sin(θ)
y' = x × sin(θ) + y × cos(θ)
```

**Example: Rotate (1, 0) by 90°**
```
x = 1, y = 0
cos(90°) = 0, sin(90°) = 1

x' = 1×0 - 0×1 = 0
y' = 1×1 + 0×0 = 1

New point: (0, 1)  ← Rotated 90° counterclockwise!
```

**Why this matters in AI:** RoPE (Rotary Position Embeddings) rotates vectors to encode position!

```
Position 1: Rotate by 10°
Position 2: Rotate by 20°
Position 3: Rotate by 30°
...

The model learns: "similar rotations = nearby positions"
```

---

## Special AI Functions

### Softmax

**What it does:** Converts numbers to probabilities (that sum to 1)

**Steps:**

```
Input numbers (logits): [2.0, 1.0, 0.1]

Step 1: Exponentiate (e^x)
e^2.0 ≈ 7.39
e^1.0 ≈ 2.72
e^0.1 ≈ 1.11

Step 2: Sum them
7.39 + 2.72 + 1.11 = 11.22

Step 3: Divide each by the sum
7.39 / 11.22 ≈ 0.66 (66%)
2.72 / 11.22 ≈ 0.24 (24%)
1.11 / 11.22 ≈ 0.10 (10%)

Output probabilities: [0.66, 0.24, 0.10]
Sum = 1.0 ✓
```

**Why this matters in AI:** Converts model's raw scores to probabilities!

**Visual:**
```
Raw scores (can be any size):
[10.5, -2.3, 5.1, 0.8]
      ↓ softmax
Probabilities (between 0 and 1, sum to 1):
[0.73, 0.00, 0.26, 0.01]
       ↑
Total = 1.00
```

**Temperature effect:**
```
Original logits: [2.0, 1.0, 0.1]

Temperature = 1.0 (normal):
Probabilities: [0.66, 0.24, 0.10]

Temperature = 0.5 (sharper):
Divide first: [4.0, 2.0, 0.2]
Probabilities: [0.84, 0.14, 0.02]  ← More confident!

Temperature = 2.0 (flatter):
Divide first: [1.0, 0.5, 0.05]
Probabilities: [0.53, 0.29, 0.18]  ← More random!
```

### ReLU (Rectified Linear Unit)

**What it does:** Keep positive numbers, zero out negative numbers

**Formula:** `ReLU(x) = max(0, x)`

**Examples:**
```
ReLU(3) = 3      (positive → keep it)
ReLU(-2) = 0     (negative → zero it)
ReLU(0) = 0      (zero → stay zero)
ReLU(5.7) = 5.7  (positive → keep it)
```

**Graph:**
```
      ReLU(x)
       │
     3 │    ╱
     2 │   ╱
     1 │  ╱
     0 ├─────→ x
  -3 -2 -1 0 1 2 3
         └─┘
       zeros out
       negatives
```

**ReLU² (ReLU squared):**
```
Formula: ReLU²(x) = max(0, x)²

ReLU²(3) = 3² = 9
ReLU²(-2) = 0² = 0
ReLU²(2) = 2² = 4
```

**Why this matters in AI:** Adds non-linearity to the model (helps it learn complex patterns)

### Cross-Entropy Loss

**What it measures:** How different are the predicted probabilities from the true answer?

**Formula:** `-log(probability of correct answer)`

**Example:**
```
True answer: "cat"

Model predictions:
"cat"  → 0.7 (70%)
"dog"  → 0.2 (20%)
"bird" → 0.1 (10%)

Loss = -log(0.7) ≈ 0.36  (lower is better)

If model predicted:
"cat"  → 0.9 (90%)  → Loss = -log(0.9) ≈ 0.11  ← Better!
"cat"  → 0.5 (50%)  → Loss = -log(0.5) ≈ 0.69  ← Worse!
"cat"  → 0.1 (10%)  → Loss = -log(0.1) ≈ 2.30  ← Much worse!
```

**Why -log?**

Because we want:
- High probability (good) → Low loss
- Low probability (bad) → High loss

And `-log` does exactly that!

```
Probability  →  -log(p)  →  Loss
1.0 (100%)   →   0.00    →  Perfect! (no loss)
0.5 (50%)    →   0.69    →  Okay
0.1 (10%)    →   2.30    →  Bad
0.01 (1%)    →   4.61    →  Very bad!
```

### RMSNorm (Root Mean Square Normalization)

**What it does:** Rescale numbers so they're not too big or too small

**Steps:**

```
Input: [2, 4, 6]

Step 1: Square each number
2² = 4
4² = 16
6² = 36

Step 2: Take the mean (average)
mean = (4 + 16 + 36) / 3 = 18.67

Step 3: Take square root
RMS = √18.67 ≈ 4.32

Step 4: Divide each number by RMS
2 / 4.32 ≈ 0.46
4 / 4.32 ≈ 0.93
6 / 4.32 ≈ 1.39

Output: [0.46, 0.93, 1.39]
```

**Formula:**
```
RMSNorm(x) = x / √(mean(x²))
```

**Why this matters in AI:** Keeps the activations stable during training

**Visual:**
```
Before RMSNorm:
Numbers can be huge:  [100, 200, 300]
      ↓ RMSNorm
After RMSNorm:
Numbers are reasonable: [0.46, 0.93, 1.39]
```

---

## Quick Reference: Notation Guide

### Greek Letters

```
α (alpha)   - often used for learning rate
β (beta)    - often used for momentum
θ (theta)   - often used for parameters/weights
ε (epsilon) - small number (like 0.0001)
σ (sigma)   - standard deviation
Σ (Sigma)   - sum
μ (mu)      - mean
∂ (partial) - partial derivative
```

### Math Symbols

```
+  addition
-  subtraction
× * ·  multiplication
÷ /  division
^  exponent (power)
√  square root
|  absolute value
≈  approximately equal
→  maps to, transforms to
∑  sum
∏  product
∂  partial derivative
```

### Subscripts and Superscripts

**Subscript (small below):** Index or identifier
```
x₁, x₂, x₃  ← different x values (x-sub-1, x-sub-2, ...)
wᵢⱼ         ← weight at row i, column j
```

**Superscript (small above):** Power or layer number
```
x²     ← x squared
x³     ← x cubed
h⁽¹⁾   ← hidden layer 1
h⁽²⁾   ← hidden layer 2
```

### Brackets and Parentheses

```
[1, 2, 3]       ← vector/list
(2 + 3) × 4     ← grouping (do this first)
⎡ 1 2 ⎤
⎣ 3 4 ⎦         ← matrix
f(x)            ← function of x
```

---

## Practice Examples

### Example 1: Calculating Attention Score

```
Query vector:  q = [1, 2]
Key vector:    k = [3, 4]

Attention score = q · k
                = (1×3) + (2×4)
                = 3 + 8
                = 11
```

### Example 2: Softmax

```
Logits: [1.0, 2.0, 3.0]

Step 1: e^x
e^1.0 ≈ 2.72
e^2.0 ≈ 7.39
e^3.0 ≈ 20.09

Step 2: Sum
2.72 + 7.39 + 20.09 = 30.20

Step 3: Divide
2.72 / 30.20 ≈ 0.09 (9%)
7.39 / 30.20 ≈ 0.24 (24%)
20.09 / 30.20 ≈ 0.67 (67%)

Probabilities: [0.09, 0.24, 0.67]
```

### Example 3: Gradient Descent Step

```
Current weight: w = 2.0
Current loss: L = 5.0
Gradient: ∂L/∂w = -3.0
Learning rate: α = 0.1

New weight = w - α × ∂L/∂w
           = 2.0 - 0.1 × (-3.0)
           = 2.0 + 0.3
           = 2.3
```

### Example 4: RMSNorm

```
Input: x = [3, 4, 5]

Step 1: Square
[9, 16, 25]

Step 2: Mean
(9 + 16 + 25) / 3 = 16.67

Step 3: Sqrt
√16.67 ≈ 4.08

Step 4: Divide
3 / 4.08 ≈ 0.74
4 / 4.08 ≈ 0.98
5 / 4.08 ≈ 1.23

Output: [0.74, 0.98, 1.23]
```

---

## Visual Summary

### The Math Journey in nanochat

```
┌─────────────────────────────────────────────┐
│           NANOCHAT MATH FLOW                │
│                                             │
│  1. Tokenization                            │
│     Text → IDs (just simple numbering)      │
│                                             │
│  2. Embeddings                              │
│     ID → Vector [x₁, x₂, ..., x₁₂₈₀]       │
│                                             │
│  3. Normalization                           │
│     RMSNorm: x / √(mean(x²))               │
│                                             │
│  4. Attention                               │
│     Q · K = attention scores                │
│     Softmax → probabilities                 │
│     Sum of (prob × V) → output              │
│                                             │
│  5. MLP                                     │
│     Matrix multiply + ReLU²                 │
│                                             │
│  6. Output                                  │
│     Logits → Softmax → Probabilities        │
│                                             │
│  7. Training                                │
│     Loss = -log(probability)                │
│     Gradient ∂L/∂w                          │
│     Update: w = w - α × ∂L/∂w              │
└─────────────────────────────────────────────┘
```

---

**Remember:** Math is just a language for describing patterns. Every formula in this guide is doing something simple - we're just using symbols to write it down precisely!

**When you see math in NANO.md:**
1. Come back here
2. Find the concept
3. Read the simple explanation
4. Look at the examples
5. Go back to NANO.md with understanding!

You've got this! 🚀
