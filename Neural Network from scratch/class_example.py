class Computador:
    def __init__(self, nome, placa_mae, ssd, ram):
        self.placa_mae = placa_mae
        self.ssd = ssd
        self.ram = ram
        self.nome = nome

    def mostra_ram(self):
        print(self.ram)

    def __str__(self):
        return f"Nome: {self.nome}\nPlaca mae: {self.placa_mae}\nSsd: {self.ssd}\nRam: {self.ram}"

    def __eq__(self, other):
        if isinstance(other, Computador):
            return self.nome == other.nome
        else:
            return self.nome == other


computador_vitor = Computador("Vitor", "msi", "kingston", "kingston")
computador_vitor2 = Computador("Vitor", "msi", "kingston", "kingston")
print(computador_vitor)
print(computador_vitor == computador_vitor2)