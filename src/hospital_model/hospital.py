from typing import Dict

from src.hospital_model.bed import BedContainer
# from src.patient import Status
from src.hospital_model.parameters import INIT_MOBILE_BED_NUMBER
from src.hospital_model.patient import Status
from src.hospital_model.utilities import print_with_title

import operator
import copy


# class BedService:
#     def __init__(self, total_bed_number: int):
#         self.total_bed_number: int = total_bed_number
#         self.available_bed_number: int = total_bed_number
#         self.bed_in_use: int = self.total_bed_number - self.available_bed_number
#
#     # Only severe patient need beds bed default
#     def arrange_bed_for_patient(self, patient_container: PatientContainer, status: Status == Status.SEVERE,
#                                 death_rate_if_no_bed: float):
#         check_type(status, Status)
#         check_type(patient_container, PatientContainer)
#         waiting_patients = patient_container.get_patient_by_have_bed(status, have_bed=False)
#         print(f"There are {len(waiting_patients)} whose status are {status}, they are waiting for bed")
#         print(f"There are {self.available_bed_number} available beds in the hospital_model")
#         if len(waiting_patients) > self.available_bed_number:
#             for i in range(self.available_bed_number):
#                 waiting_patients[i].have_bed = True
#             self.available_bed_number = 0
#             self.bed_in_use = self.total_bed_number
#         else:
#             for patient in waiting_patients:
#                 patient.have_bed = True
#             self.available_bed_number -= len(waiting_patients)
#             self.bed_in_use: int = self.total_bed_number - self.available_bed_number
#         print(
#             f"After arrange beds, there are {len(patient_container.get_patient_by_have_bed(status.SEVERE))}"
#             f" severe patients in total who have beds now")
#         # Set patient who do not have bed to dead
#         patient_without_bed = patient_container.get_patient_by_have_bed(status=Status.SEVERE, have_bed=False)
#         dead_patients_count = int(len(patient_without_bed) * death_rate_if_no_bed)
#         for i in range(dead_patients_count):
#             patient_without_bed[i].status = Status.DEAD
#         print(
#             f"{dead_patients_count} patients are dead because of no available beds ")
#
#         return dead_patients_count
#
#     def get_available(self):
#         return self.available_bed_number
#
#     def get_bed_in_use(self):
#         return self.bed_in_use
#
#     def free_up_bed(self, bed_number: int):
#         assert bed_number <= self.bed_in_use
#         assert bed_number <= self.total_bed_number
#         print(f"{bed_number} beds are freed up")
#         self.available_bed_number += bed_number
#         self.bed_in_use -= bed_number
#
#     def print_summary(self):
#         print("***************************BED SUMMARY*******************************")
#         print(f"Total bed number: {self.total_bed_number}")
#         print(f"Available bed number: {self.available_bed_number}")
#         print(f"Bed in use: {self.bed_in_use}")
#         print("*********************************************************************")
#
#     def get_summary(self):
#         return self.total_bed_number, self.available_bed_number, self.bed_in_use
#
#
# class VentilatorService:
#     def __init__(self, ventilator_supply: float, success_rate: float, death_rate_when_severe: float):
#         self.ventilator_supply = ventilator_supply
#         self.success_rate = success_rate
#         self.death_rate_when_severe = death_rate_when_severe
#
#     def apply_patient(self, patient_container: PatientContainer, bed: BedService):
#         bed_in_use = bed.get_bed_in_use()
#         apply_patient_count = int(bed_in_use * (self.success_rate - self.death_rate_when_severe) *
#                                   self.ventilator_supply)
#         fail_to_apply_count = bed_in_use - apply_patient_count
#         print(f"{apply_patient_count} patients will survive because ventilator is given.")
#         print(f"{fail_to_apply_count} patients will die because ventilator is not successfully given.")
#         waiting_patient = patient_container.get_patient_by_have_bed(status=Status.SEVERE, have_bed=True)
#         # check if they all have bed, below piece of code is ugly!!!
#         for patient in waiting_patient:
#             assert patient.have_bed is True
#             patient.served_days += 1
#         for i in range(fail_to_apply_count):
#             waiting_patient[i].status = Status.DEAD
#         # free up bed
#         bed.free_up_bed(fail_to_apply_count)
#         return apply_patient_count, fail_to_apply_count
#
#
# class CureService:
#     def __init__(self, days_to_cure: int):
#         self.days_to_cure = days_to_cure
#
#     def cure_patient(self, patient_container: PatientContainer, bed: BedService):
#         waiting_patient = patient_container.get_patient_by_have_bed(status=Status.SEVERE)
#         cured_counter = 0
#         for patient in waiting_patient:
#             assert patient.have_bed is True
#             if patient.served_days == self.days_to_cure:
#                 patient.status = Status.HEALTHY
#                 cured_counter += 1
#         print(f"{cured_counter} patients are cured now!!")
#         bed.free_up_bed(cured_counter)


class HospitalModel:
    # TODO: assign different bed numbers to hospitals
    def __init__(self, init_bed_number: int):
        print("HospitalModel initiated....")
        self.mobile_hospital = MobileHospital(init_bed_number)  # 方舱医院初始化

    # TODO: agree on the handshake data structure
    # Assume following data structure
    # {"Mild":[0,2,3,4],"Severe":[1,2,3]}
    def receive_patient(self, patients: Dict):
        print("Patient received...")
        msg = ""
        for key, value in patients.items():
            msg += f"{key}:\n"
            if len(value) == 0:
                msg += f"\tThere is no patient as {key}\n"
            else:
                for i in range(len(value)):
                    msg += f"\t{i + 1:2} days: {value[i]}\n"
        print_with_title(title="HospitalModel summary", body=msg)
        return self.send_to_hospital(patients)

    def send_to_hospital(self, patients: Dict):
        """
        Dispatch patients to different hospitals according to their severity
        :param patients: dictionary ie {"mild": [0, 2, 3, 4], "severe": [1, 2, 3]}
        :return:
        {"mild":   {"accepted":[0, 1, 2, 3], "rejected":[0, 1, 1, 1]},
         "severe": {"accepted":[0, 0, 2, 1], "rejected":[0, 1, 1, 0]}}
        """
        # mild are sent to mobile hospital_model
        result_dict = {}
        if 'mild' in patients.keys():
            accepted, rejected = self.mobile_hospital.assign_bed(patients['mild'])
            result_dict['mild'] = {"accepted": accepted,
                                   "rejected": rejected}
        # TODO: severe patients are sent to designated hospitals
        if 'severe' in patients.keys():
            print("Designated hospital model not implemented")
        return result_dict


class Hospital:
    def __init__(self, init_bed_number: int):
        print(f"{self.__class__.__name__} initiated...")
        self.bed_container = BedContainer(hospital_name=self.__class__.__name__).add_bed(init_bed_number)


# 方舱
class MobileHospital(Hospital):
    def __init__(self, init_bed_number: int):
        super().__init__(init_bed_number)

    def assign_bed(self, patients: list) -> tuple:
        """
        This piece of logic is used to assign bed to the patients according to current bed availability and their courses
        :param patients: list of patients, index represents course, value represents number of patients
        :return: list of patients being taken, list oa patients being rejected
        Example:
            patients_accepted = [0, 0, 0, 3]
            patients_rejected = [0, 2, 3, 1]
        """
        print(f"{sum(patients)} patients have been sent to {self.__class__.__name__}")
        available_bed_count = len(self.get_available_beds())
        patients_rejected = copy.copy(patients)  # keep the original copy
        for i in range(len(patients_rejected)):
            if available_bed_count <= int(patients_rejected[-i - 1]):
                patients_rejected[-i - 1] -= available_bed_count
                available_bed_count = 0
            else:
                available_bed_count -= patients_rejected[-i - 1]
                patients_rejected[-i - 1] = 0
        # item by item subtraction to get patients assigned to beds (accepted patients)
        patients_accepted = list(map(operator.sub, patients, patients_rejected))
        # print(available_bed_count, patients_accepted, patients_rejected)
        self.bed_container.allocate_beds(patients_list=patients_accepted, status=Status.MILD)

        print(f"Patients accepted: {patients_accepted}")
        print(f"Patients rejected: {patients_rejected}")
        print(f"Available beds: {len(self.get_available_beds())}")
        return patients_accepted, patients_rejected

    def get_available_beds(self):
        return self.bed_container.get_empty_beds()


# 定点医院
class DesignatedHospitals(Hospital):
    def __int__(self, init_bed_number: int):
        super().__init__(init_bed_number)

    def assign_bed(self, patients: list):
        pass


if __name__ == '__main__':
    model = HospitalModel(6)
    model.receive_patient(patients={"mild": [0, 2, 3, 4], "severe": [1, 2, 3], "whatever": []})
