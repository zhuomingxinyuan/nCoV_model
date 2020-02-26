from enum import Enum


# from src.utilities import check_type


class Status(Enum):
    CURED = 'CURED'
    MILD = 'MILD'
    SEVERE = 'SEVERE'
    DEAD = 'DEAD'


class Patient:
    def __init__(self, status: Status, course: int, next_status, next_status_day:int):
        self.age = None  # Not used
        self.status: Status = status
        self.course = course
        self.next_status = next_status
        self.next_status_day = next_status_day
        # self.course_disease: int = 0
        # self.have_bed: bool = False
        # self.served_days: int = 0


#     def set_status(self, new_status: Status):
#         check_type(new_status, Status)
#         self.status = new_status
#
#

def create_n_patients(n: int, course: int, status: Status):
    print(f"{n} patients are created")
    return [Patient(course=course, status=status) for i in range(n)]


#
# class PatientContainer:
#     def __init__(self):
#         self._d = []
#
#     def add_patient(self, new_patient: Patient):
#         check_type(new_patient, Patient)
#         self._d.append(new_patient)
#
#     def create_n_patient(self, n: int, status=Status.HEALTHY):
#         for i in range(n):
#             self.add_patient(Patient(status))
#
#     def get_patient_by_status(self, status: Status):
#         check_type(status, Status)
#         return [patient for patient in self._d if patient.status == status]
#
#     def get_patient_by_have_bed(self, status: Status, have_bed: bool = True):
#         return [patient for patient in self._d if (patient.have_bed is have_bed) and (patient.status == status)]
#
#     def print_summary(self):
#         print("************************PATIENT SUMMARY*******************************")
#         print(f"There are {len(self._d)} patients.")
#         for status in Status:
#             print(
#                 f"There are {len(self.get_patient_by_status(status))} whose status are {status}")
#         print("**********************************************************************")
#
#     def get_summary(self):
#         return [len(self.get_patient_by_status(status)) for status in Status]

if __name__ == '__main__':
    for patient in create_n_patients(status=Status.MILD, course=2, n=3):
        print(patient, patient.course, patient.status, )
