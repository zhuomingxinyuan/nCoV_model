from src.hospital_model.patient import Patient, Status


class BedContainer:
    def __init__(self, hospital_name: str):
        print(f"Bed container for {hospital_name} is initiated...")
        self.beds = []

    def add_bed(self, n: int):
        for i in range(n):
            self.beds.append(Bed())
        print(f"{n} beds has been added")
        return self

    def get_empty_beds(self):
        return [bed for bed in self.beds if bed.patient is None]

    def allocate_beds(self, patients_list: list, status: Status):
        for index in range(len(patients_list)):
            for patient in range(patients_list[index]):
                new_patient = Patient(course=index + 1, status=status)
                self.get_empty_beds()[0].assign_patient(new_patient)
        print(f"{sum(patients_list)} patients have been allocated with beds.")

    import numpy as np
    # assume severe/total = 0.1
    # 5 parameters
    # severe/total -> severe/cured
    # mild1 -> severe -- normal
    # mild2 -> severe -- normal
    # mild1 -> cured --normal
    # mild2 -> cured --normal
    Patient(status=Status.MILD, course=2, next_status=Status.SEVERE, next_status_days=np.random.normal(loc=mean,))
    Patient(status=Status.MILD, course=2, next_status=Status.CURED)
    Patient(status=Status.MILD, course=2, next_status=Status.CURED)
    Patient(status=Status.MILD, course=2, next_status=Status.CURED)
    Patient(status=Status.MILD, course=2, next_status=Status.CURED)
    Patient(status=Status.MILD, course=2, next_status=Status.CURED)
    Patient(status=Status.MILD, course=2, next_status=Status.CURED)
    Patient(status=Status.MILD, course=2, next_status=Status.CURED)
    Patient(status=Status.MILD, course=2, next_status=Status.CURED)
    Patient(status=Status.MILD, course=2, next_status=Status.CURED)

    # def get_patient_by_status(self, status: Status):
    #     return [bed for bed in self.beds if bed.patient.status == status]


class Bed:
    def __init__(self):
        self.patient = None

    def get_patent(self):
        return self.patient

    def assign_patient(self, new_patient: Patient):
        self.patient = new_patient


if __name__ == '__main__':
    print(len(BedContainer("test_hospital").add_bed(10).get_empty_beds()))
