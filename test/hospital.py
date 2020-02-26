from src.hospital_model.hospital import MobileHospital, HospitalModel
from src.hospital_model.utilities import print_with_title

print_with_title(title="TEST CASE 1")
model = HospitalModel(2)
expected_result = {"accepted": [0, 0, 0, 2], "rejected": [0, 2, 3, 2]}
assert model.receive_patient(patients={"mild": [0, 2, 3, 4], "severe": [1, 2, 3], "whatever": []})[
           'mild'] == expected_result

print_with_title(title="TEST CASE 2")
model = HospitalModel(4)
expected_result = {"accepted": [0, 0, 0, 4], "rejected": [0, 2, 3, 0]}
assert model.receive_patient(patients={"mild": [0, 2, 3, 4], "severe": [1, 2, 3], "whatever": []})[
           'mild'] == expected_result

print_with_title(title="TEST CASE 3")
model = HospitalModel(10)
expected_result = {"accepted": [0, 2, 3, 4], "rejected": [0, 0, 0, 0]}
assert model.receive_patient(patients={"mild": [0, 2, 3, 4], "severe": [1, 2, 3], "whatever": []})[
           'mild'] == expected_result

print_with_title(title="TEST CASE 4")
model = HospitalModel(20)
expected_result = {"accepted": [0, 2, 3, 4], "rejected": [0, 0, 0, 0]}
assert model.receive_patient(patients={"mild": [0, 2, 3, 4], "severe": [1, 2, 3], "whatever": []})[
           'mild'] == expected_result