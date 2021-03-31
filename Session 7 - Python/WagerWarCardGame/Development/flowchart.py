from pyflowchart import Flowchart
with open('Development\WWShuffleDeck.py') as f:
    code = f.read()

fc = Flowchart.from_code(code)
print(fc.flowchart())

# output flowchart code.
