
colors = [
          "1B2A34",
          "1E5AA8",
          "00852B",
          "069D9F",
          "B40000",
          "D05098",
          "D3359D",
          "543324",
          "8A928D",
          "545955",
          "97CBD9",
          "58AB41",
          "00AAA4",
          "F06D61",
          "F6A9BB",
          "FAC80A",
          "F4F4F4",
          "ADD9A8",
          "FFD67F",
          "B0A06F",
          "AFBED6",
          "E5DFD3",
          "671F81",
          "0E3E9A",
          "D67923",
          "901F76",
          "0A1327",
          "6A7944",
          "3E95B6",
          "D4E5AB",
          "F17880",
          "564E9D",
          "A47624",
          "AC8247",
          "D60026",
          "945148",
          "AD6140",
          "56472F",
          "FF9494"
        ]

def hex2rgb(hex):
    rgb_tuple = tuple(int(hex[i:i+len(hex)//3], 16) for i in range(0, len(hex), len(hex)//3))
    print(rgb_tuple)
    return ''.join(str(c) for c in rgb_tuple) + ''

hex2rgb("FF9494")

colors = ['#' + c for c in colors]

import matplotlib.pyplot as plt
for x, color in enumerate(colors):
    plt.plot(x + 200, marker='o', color=color, label=color, markersize=10)
plt.legend()
plt.show()