"""
latent-collab / problems.py
----------------------------
20 arithmetic word problems with known numeric answers.
Each entry is a dict: { "question": str, "answer": float }

Designed to be:
  • Self-contained (no external knowledge needed)
  • Verifiable programmatically (single numeric answer)
  • Varied in operation type (×, ÷, %, fractions, rates, compound)
"""

PROBLEMS = [
    {
        "question": (
            "A bakery produces 144 cookies per batch. "
            "They bake 3 batches in the morning and 2 batches in the afternoon. "
            "How many cookies do they produce in total?"
        ),
        "answer": 720.0,
    },
    {
        "question": (
            "Tom has $85.00. He spends $23.50 on groceries and $12.75 on a book. "
            "How much money does he have left?"
        ),
        "answer": 48.75,
    },
    {
        "question": (
            "A train travels at 60 km/h. "
            "How many kilometers does it cover in 2.5 hours?"
        ),
        "answer": 150.0,
    },
    {
        "question": (
            "A rectangular garden is 15 metres long and 8 metres wide. "
            "What is its area in square metres?"
        ),
        "answer": 120.0,
    },
    {
        "question": (
            "Sarah reads 45 pages per day. "
            "How many pages will she read in exactly 3 weeks?"
        ),
        "answer": 945.0,
    },
    {
        "question": (
            "Apples cost $0.75 each. "
            "How much do 12 apples cost in total?"
        ),
        "answer": 9.0,
    },
    {
        "question": (
            "A water tank holds 500 litres when full. "
            "It is currently 40 % full. "
            "How many litres of water are in the tank?"
        ),
        "answer": 200.0,
    },
    {
        "question": (
            "A shirt has an original price of $35. "
            "It is on sale with a 20 % discount. "
            "What is the sale price?"
        ),
        "answer": 28.0,
    },
    {
        "question": (
            "A class has 32 students. "
            "Exactly 3/8 of them play soccer. "
            "How many students play soccer?"
        ),
        "answer": 12.0,
    },
    {
        "question": (
            "A car consumes 8 litres of fuel per 100 km. "
            "How many litres are needed for a 350 km journey?"
        ),
        "answer": 28.0,
    },
    {
        "question": (
            "Maria earns $18 per hour and works 40 hours per week. "
            "How much does she earn in one week?"
        ),
        "answer": 720.0,
    },
    {
        "question": (
            "A 48-metre rope is cut into pieces of 1.5 metres each. "
            "How many pieces are there?"
        ),
        "answer": 32.0,
    },
    {
        "question": (
            "A box contains 240 chocolates. "
            "Five friends share them equally. "
            "How many chocolates does each person receive?"
        ),
        "answer": 48.0,
    },
    {
        "question": (
            "The temperature at midnight was -5 °C. "
            "By noon it had risen to 22 °C. "
            "By how many degrees did the temperature rise?"
        ),
        "answer": 27.0,
    },
    {
        "question": (
            "A swimming pool is 25 metres long. "
            "A swimmer completes 30 lengths. "
            "How many metres does she swim in total?"
        ),
        "answer": 750.0,
    },
    {
        "question": (
            "A farmer collects 180 eggs and packs them into boxes of 12. "
            "How many full boxes does he fill?"
        ),
        "answer": 15.0,
    },
    {
        "question": (
            "Seven workers can build a wall in 12 days. "
            "How many workers are needed to build the same wall in 4 days, "
            "assuming each worker works at the same rate?"
        ),
        "answer": 21.0,
    },
    {
        "question": (
            "A principal of $2 000 earns simple interest at 5 % per year. "
            "How much interest is earned after 3 years?"
        ),
        "answer": 300.0,
    },
    {
        "question": (
            "A recipe uses 2.5 cups of flour to make 20 cookies. "
            "How many cups of flour are needed to make 60 cookies?"
        ),
        "answer": 7.5,
    },
    {
        "question": (
            "A shop sells pens at $1.20 each and notebooks at $3.50 each. "
            "Alice buys 4 pens and 3 notebooks. "
            "How much does she pay in total?"
        ),
        "answer": 15.3,
    },
]
