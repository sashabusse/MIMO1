class TaskCfg:
    def __init__(self, packet_cnt, sc_cnt, student_id):
        self.packet_cnt = packet_cnt
        self.sc_cnt = sc_cnt

        # Depending of Student ID in this task, each one has personal sub-carrier offset,
        # [ Student# * 12 + 1  ... Student# * 12 + 1 + L ]
        # REMEMBER(!)     L < K - 12* StudentID
        # ------------------------------------------------------------------------------
        self.student_id = student_id
        self.sc0 = self.student_id * 12 + 1
        self.sc1 = self.student_id * 12 + 1 + self.sc_cnt
        self.time_offset = 2 * self.student_id

