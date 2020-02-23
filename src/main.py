# from src.patient import PatientContainer, Status
# from src.hospital import BedService, VentilatorService, CureService
# import matplotlib.pyplot as plt
# from src.utilities import Result
#
# # TODO: floor to int when there is a float
# total_bed_number = 500  # 病床数
# daily_new_severe_patient = 50  # 每日新增重症患者数量
# ventilator_supply_rate = 0.98  # 呼吸机供应比率
# ventilator_success_rate = 0.98  # 呼吸机救治率
# death_rate_when_severe = 0.01  # 重症死亡率（呼吸机无效）
# death_rate_if_no_bed = 0.9  # 无床位重症病人死亡率
# days_to_cure = 10  # 重症病人在呼吸机支持下康复所需天数
# initial_severe_patient_number = 500  # 起始重症病人数
# simulation_steps = 50  # 模拟天数
#
# patient_container = PatientContainer()
# bed = BedService(total_bed_number=total_bed_number)
#
# patient_container.create_n_patient(initial_severe_patient_number, Status.SEVERE)
# patient_summary_result = Result(name='patient summary', cols=[status.value for status in Status]).init_df()
# death_summary_result = Result(name='death increase summary',
#                               cols=["Total death", "Death due to bed", "Death due to vent"]).init_df()
# bed_summary_result = Result(name='bed summary', cols=["Total bed", "Available bed", "Bed in use"]).init_df()
#
# for i in range(simulation_steps):
#     print(f"____________________________________________{i}_________________________________________________")
#     patient_container.create_n_patient(daily_new_severe_patient, Status.SEVERE)
#     patient_container.print_summary()
#     bed.print_summary()
#     dead_patient_count_bed = bed.arrange_bed_for_patient(patient_container, status=Status.SEVERE,
#                                                          death_rate_if_no_bed=death_rate_if_no_bed)
#
#     print(f"There are {bed.get_available()} available bed now")
#
#     ventilator = VentilatorService(ventilator_supply=ventilator_supply_rate, success_rate=ventilator_success_rate,
#                                    death_rate_when_severe=death_rate_when_severe)
#     _, dead_patient_count_vent = ventilator.apply_patient(patient_container, bed)
#
#     cure = CureService(days_to_cure=days_to_cure)
#     cure.cure_patient(patient_container, bed)
#
#     patient_container.print_summary()
#     bed.print_summary()
#
#     patient_summary_result.add_row(idx=i + 1, row=patient_container.get_summary())
#     death_summary_result.add_row(idx=i + 1, row=[dead_patient_count_bed + dead_patient_count_vent,
#                                                  dead_patient_count_bed,
#                                                  dead_patient_count_vent])
#     bed_summary_result.add_row(idx=i + 1, row=list(bed.get_summary()))
#     print(f"____________________________________________{i} end_____________________________________________")
#
# print(patient_summary_result.get_df())
# print(death_summary_result.get_df())
# print(bed_summary_result.get_df())
#
# fig, axes = plt.subplots(nrows=3, ncols=1)
#
# patient_summary_result.get_df().plot(title=patient_summary_result.name, ax=axes[0, ])
# death_summary_result.get_df().plot(title=death_summary_result.name, ax=axes[1, ])
# bed_summary_result.get_df().plot(title=bed_summary_result.name, ax=axes[2, ])
# plt.show()
