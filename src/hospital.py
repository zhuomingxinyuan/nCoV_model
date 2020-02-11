from src.patient import PatientContainer, Status
from src.utilities import check_type


class BedService:
    def __init__(self, total_bed_number: int):
        self.total_bed_number: int = total_bed_number
        self.available_bed_number: int = total_bed_number
        self.bed_in_use: int = self.total_bed_number - self.available_bed_number

    # Only severe patient need beds bed default
    def arrange_bed_for_patient(self, patient_container: PatientContainer, status: Status == Status.SEVERE,
                                death_rate_if_no_bed: float):
        check_type(status, Status)
        check_type(patient_container, PatientContainer)
        waiting_patients = patient_container.get_patient_by_have_bed(status, have_bed=False)
        print(f"There are {len(waiting_patients)} whose status are {status}, they are waiting for bed")
        print(f"There are {self.available_bed_number} available beds in the hospital")
        if len(waiting_patients) > self.available_bed_number:
            for i in range(self.available_bed_number):
                waiting_patients[i].have_bed = True
            self.available_bed_number = 0
            self.bed_in_use = self.total_bed_number
        else:
            for patient in waiting_patients:
                patient.have_bed = True
            self.available_bed_number -= len(waiting_patients)
            self.bed_in_use: int = self.total_bed_number - self.available_bed_number
        print(
            f"After arrange beds, there are {len(patient_container.get_patient_by_have_bed(status.SEVERE))}"
            f" severe patients in total who have beds now")
        # Set patient who do not have bed to dead
        patient_without_bed = patient_container.get_patient_by_have_bed(status=Status.SEVERE, have_bed=False)
        dead_patients_count = int(len(patient_without_bed) * death_rate_if_no_bed)
        for i in range(dead_patients_count):
            patient_without_bed[i].status = Status.DEAD
        print(
            f"{dead_patients_count} patients are dead because of no available beds ")

        return dead_patients_count

    def get_available(self):
        return self.available_bed_number

    def get_bed_in_use(self):
        return self.bed_in_use

    def free_up_bed(self, bed_number: int):
        assert bed_number <= self.bed_in_use
        assert bed_number <= self.total_bed_number
        print(f"{bed_number} beds are freed up")
        self.available_bed_number += bed_number
        self.bed_in_use -= bed_number

    def print_summary(self):
        print("***************************BED SUMMARY*******************************")
        print(f"Total bed number: {self.total_bed_number}")
        print(f"Available bed number: {self.available_bed_number}")
        print(f"Bed in use: {self.bed_in_use}")
        print("*********************************************************************")

    def get_summary(self):
        return self.total_bed_number, self.available_bed_number, self.bed_in_use


class VentilatorService:
    def __init__(self, ventilator_supply: float, success_rate: float, death_rate_when_severe: float):
        self.ventilator_supply = ventilator_supply
        self.success_rate = success_rate
        self.death_rate_when_severe = death_rate_when_severe

    def apply_patient(self, patient_container: PatientContainer, bed: BedService):
        bed_in_use = bed.get_bed_in_use()
        apply_patient_count = int(bed_in_use * (self.success_rate - self.death_rate_when_severe) *
                                  self.ventilator_supply)
        fail_to_apply_count = bed_in_use - apply_patient_count
        print(f"{apply_patient_count} patients will survive because ventilator is given.")
        print(f"{fail_to_apply_count} patients will die because ventilator is not successfully given.")
        waiting_patient = patient_container.get_patient_by_have_bed(status=Status.SEVERE, have_bed=True)
        # check if they all have bed, below piece of code is ugly!!!
        for patient in waiting_patient:
            assert patient.have_bed is True
            patient.served_days += 1
        for i in range(fail_to_apply_count):
            waiting_patient[i].status = Status.DEAD
        # free up bed
        bed.free_up_bed(fail_to_apply_count)
        return apply_patient_count, fail_to_apply_count


class CureService:
    def __init__(self, days_to_cure: int):
        self.days_to_cure = days_to_cure

    def cure_patient(self, patient_container: PatientContainer, bed: BedService):
        waiting_patient = patient_container.get_patient_by_have_bed(status=Status.SEVERE)
        cured_counter = 0
        for patient in waiting_patient:
            assert patient.have_bed is True
            if patient.served_days == self.days_to_cure:
                patient.status = Status.HEALTHY
                cured_counter += 1
        print(f"{cured_counter} patients are cured now!!")
        bed.free_up_bed(cured_counter)
