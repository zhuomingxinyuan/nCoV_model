class BedContainer:
    def __init__(self, hospital_name: str):
        print(f"Bed container for {hospital_name} is initiated...")
        self.beds = []

    def add_bed(self, n: int):
        for i in range(n):
            self.beds.append(Bed())
        print(f"{n} beds has been added")
        return self

    def get_empty_bed(self):
        return [bed for bed in self.beds if bed.patient is None]


class Bed:
    def __init__(self):
        self.patient = None

    def get_patent(self):
        return self.patient

    # def assign_patient(self, new_patient):
    #     self.patient = new_patient


if __name__ == '__main__':
    print(len(BedContainer("test_hospital").add_bed(10).get_empty_bed()))
