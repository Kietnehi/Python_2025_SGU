# qlsv_simple_refactored_noted.py
# Bản đơn giản, có:
# - Kế thừa (Undergraduate, Graduate kế thừa StudentBase)
# - Đa hình (student_type, tuition, status mỗi lớp xử lý khác nhau)
# - "Trừu tượng" bằng NotImplementedError
# - Đóng gói nhẹ bằng thuộc tính _id, _name, ... + @property

import json

# ========================
# LỚP CƠ SỞ (giả "abstract")
# ========================
class StudentBase:
    # Bảng quy đổi điểm chữ -> điểm số
    LETTER_POINTS = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
    VALID_LETTERS = set(LETTER_POINTS.keys())

    def __init__(self, sid, name, year):
        # Ép kiểu + strip để tránh lỗi dữ liệu nhập linh tinh
        sid = str(sid).strip()
        name = str(name).strip()
        year = int(year)

        # Một chút kiểm tra dữ liệu (validation)
        if not sid:
            raise ValueError("ID không được rỗng")
        if not name:
            raise ValueError("Tên không được rỗng")
        if not (1 <= year <= 6):
            raise ValueError("Năm phải trong khoảng 1..6")

        # Thuộc tính "protected": convention _ten_bien (không phải private tuyệt đối)
        self._id = sid
        self._name = name
        self._year = year
        # Danh sách môn học: mỗi phần tử là 1 dict {name, credits, letter}
        self._courses = []

    # ---- property để truy cập an toàn hơn (đóng gói nhẹ) ----
    @property
    def id(self):
        # Cho phép đọc s.id thay vì s._id
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def year(self):
        return self._year

    @property
    def courses(self):
        # Trả về bản copy để tránh bị sửa trực tiếp từ bên ngoài
        return list(self._courses)

    # ---- hành vi chung cho mọi loại sinh viên ----
    def add_course(self, name, credits, letter):
        # Thêm 1 môn học vào danh sách
        name = name.strip()
        if not name:
            raise ValueError("Tên môn không được rỗng")

        letter = letter.strip().upper()
        if letter not in self.VALID_LETTERS:
            raise ValueError("Điểm phải là A/B/C/D/F")

        credits = int(credits)
        if credits <= 0:
            raise ValueError("Tín chỉ phải > 0")

        self._courses.append({
            "name": name,
            "credits": credits,
            "letter": letter
        })

    def gpa(self):
        """
        Tính GPA theo công thức:
        GPA = (tổng (điểm chữ quy đổi * tín chỉ)) / (tổng tín chỉ)
        Nếu chưa có môn nào -> GPA = 0.0
        """
        if not self._courses:
            return 0.0

        total_p = 0.0  # tổng điểm quy đổi * tín chỉ
        total_c = 0    # tổng tín chỉ

        for c in self._courses:
            total_p += self.LETTER_POINTS[c["letter"]] * c["credits"]
            total_c += c["credits"]

        return round(total_p / total_c, 2) if total_c else 0.0

    # ---- đa hình / "trừu tượng" đơn giản ----
    # Các hàm này không có body ở lớp cha, bắt buộc lớp con override lại.
    # Nếu lớp con không override mà gọi -> sẽ bị NotImplementedError.
    def student_type(self):
        raise NotImplementedError

    def tuition(self):
        raise NotImplementedError

    def status(self):
        raise NotImplementedError

    # ---- tiện ích xuất/nhập (dùng cho lưu JSON) ----
    def to_dict(self):
        """
        Chuyển 1 object Student thành dict để chuẩn bị dump sang JSON.
        Lưu cả type để khi load lên biết cần tạo đối tượng Undergraduate hay Graduate.
        """
        return {
            "type": self.student_type(),
            "id": self._id,
            "name": self._name,
            "year": self._year,
            "courses": list(self._courses),
        }

    @staticmethod
    def from_dict(d):
        """
        Hàm "factory" tạo object từ dict (đọc JSON lên).
        Tùy theo field 'type' mà tạo Undergraduate hay Graduate.
        """
        t = (d.get("type") or "").lower()
        sid = d["id"]
        name = d["name"]
        year = d["year"]

        if t == "undergraduate":
            s = Undergraduate(sid, name, year)
        elif t == "graduate":
            s = Graduate(sid, name, year, d.get("advisor", ""))
        else:
            raise ValueError("Loại SV không hợp lệ")

        # Thêm các môn học đã lưu
        for c in d.get("courses", []):
            s.add_course(c["name"], c["credits"], c["letter"])
        return s


# ========================
# LỚP CON: ĐẠI HỌC
# ========================
class Undergraduate(StudentBase):
    # Học phí / 1 tín chỉ
    BASE_PER_CREDIT = 350_000
    SENIOR_FACTOR = 1.10  # Năm 4 trở lên bị phụ thu 10%

    def student_type(self):
        # Đa hình: mỗi lớp trả về kiểu khác nhau
        return "Undergraduate"

    def tuition(self):
        """
        Học phí = tổng_tín_chỉ * BASE_PER_CREDIT
        Nếu năm học >= 4 -> * 1.10 (phụ 10%)
        """
        credits = sum(c["credits"] for c in self._courses)
        fee = credits * self.BASE_PER_CREDIT
        if self._year >= 4:
            fee *= self.SENIOR_FACTOR
        return round(fee, 2)

    def status(self):
        """
        Xếp loại dựa vào GPA:
        - >= 3.2 : Honors
        - >= 2.0 : Good
        - < 2.0  : Probation
        """
        g = self.gpa()
        if g >= 3.2:
            return "Honors (ĐH)"
        if g >= 2.0:
            return "Good (ĐH)"
        return "Probation (ĐH)"


# ========================
# LỚP CON: CAO HỌC
# ========================
class Graduate(StudentBase):
    # Học phí / tín chỉ cao học cao hơn
    BASE_PER_CREDIT = 600_000
    MERIT_DISCOUNT = 0.95  # Nếu GPA >= 3.5 giảm 5% (nhân 0.95)

    def __init__(self, sid, name, year, advisor=""):
        # Gọi __init__ của lớp cha để set id, name, year, courses
        super().__init__(sid, name, year)
        # Người hướng dẫn luận văn
        self._advisor = advisor.strip()

    @property
    def advisor(self):
        return self._advisor

    def student_type(self):
        return "Graduate"

    def tuition(self):
        """
        Học phí = tổng_tín_chỉ * BASE_PER_CREDIT
        Nếu GPA >= 3.5 và có ít nhất 1 tín chỉ -> giảm 5%.
        """
        credits = sum(c["credits"] for c in self._courses)
        fee = credits * self.BASE_PER_CREDIT
        if self.gpa() >= 3.5 and credits > 0:
            fee *= self.MERIT_DISCOUNT
        return round(fee, 2)

    def status(self):
        """
        Xếp loại cao học:
        - GPA >= 3.5 : Distinction
        - GPA >= 2.5 : Good
        - < 2.5      : Probation
        """
        g = self.gpa()
        if g >= 3.5:
            return "Distinction (CH)"
        if g >= 2.5:
            return "Good (CH)"
        return "Probation (CH)"

    def to_dict(self):
        """
        Ghi đè to_dict để lưu thêm 'advisor'.
        Vẫn gọi super().to_dict() để không lặp code.
        """
        d = super().to_dict()
        d["advisor"] = self._advisor
        return d


# ========================
# QUẢN LÝ DANH SÁCH SINH VIÊN
# ========================
class StudentManager:
    def __init__(self):
        # Danh sách toàn bộ sinh viên (Undergraduate + Graduate)
        self._list = []

    # --------- CRUD (Create - Read - Update - Delete) ----------
    def add(self, s: StudentBase):
        """
        Thêm 1 sinh viên.
        - Kiểm tra kiểu: phải là StudentBase hoặc lớp con.
        - Kiểm tra trùng ID.
        """
        if not isinstance(s, StudentBase):
            raise TypeError("Chỉ được thêm đối tượng kiểu StudentBase hoặc kế thừa")
        if self.find_by_id(s.id):
            raise ValueError("ID đã tồn tại")
        self._list.append(s)

    def remove(self, sid):
        """
        Xóa sinh viên theo ID.
        Trả về True nếu xóa được, False nếu không tìm thấy.
        """
        s = self.find_by_id(sid)
        if s:
            self._list.remove(s)
            return True
        return False

    def find_by_id(self, sid):
        """
        Tìm sinh viên theo ID.
        Trả về object sinh viên hoặc None nếu không thấy.
        """
        sid = sid.strip()
        for s in self._list:
            if s.id == sid:
                return s
        return None

    def search_by_name(self, kw):
        """
        Tìm kiếm gần đúng theo tên (chứa từ khóa, không phân biệt hoa thường).
        Trả về list sinh viên phù hợp.
        """
        kw = kw.strip().lower()
        return [s for s in self._list if kw in s.name.lower()]

    # --------- Sắp xếp ----------
    def sort_by_gpa(self, desc=True):
        """
        Trả về list mới (không sửa _list gốc)
        được sắp xếp theo GPA giảm dần (mặc định).
        """
        return sorted(self._list, key=lambda x: x.gpa(), reverse=desc)

    def sort_by_name(self):
        """
        Trả về list mới sắp xếp theo tên (a -> z).
        """
        return sorted(self._list, key=lambda x: x.name.lower())

    # --------- Báo cáo dạng list[dict] để in bảng ----------
    def report_rows(self, data=None):
        """
        Chuyển danh sách sinh viên thành list dict đơn giản:
        mỗi dict chứa các field cần in bảng: id, name, type, year, gpa, ...
        """
        data = data if data is not None else self._list
        rows = []
        for s in data:
            rows.append({
                "id": s.id,
                "name": s.name,
                "type": s.student_type(),
                "year": s.year,
                "gpa": s.gpa(),
                "tuition": s.tuition(),
                "status": s.status(),
                "num_courses": len(s.courses),
            })
        return rows

    # --------- In bảng ASCII đơn giản ----------
    def print_table(self, rows):
        """
        In bảng dạng ASCII:
        +------+--------+ ...
        | ID   | NAME   | ...
        """
        if not rows:
            print("Không có dữ liệu.")
            return

        keys = ["id", "name", "type", "year", "gpa", "tuition", "status", "num_courses"]

        # Tính chiều rộng mỗi cột = max(len(tên cột), len dữ liệu)
        widths = {
            k: max(len(k), max(len(str(r[k])) for r in rows))
            for k in keys
        }

        def line(ch="-"):
            # In 1 dòng kẻ với ký tự ch ( - hoặc = )
            print("+", end="")
            for k in keys:
                print(ch * (widths[k] + 2) + "+", end="")
            print()

        # header
        line("=")
        print("| " + " | ".join(k.upper().ljust(widths[k]) for k in keys) + " |")
        line("=")
        # rows
        for r in rows:
            print("| " + " | ".join(str(r[k]).ljust(widths[k]) for k in keys) + " |")
        line("=")

    # --------- Lưu/đọc JSON ----------
    def save(self, path):
        """
        Lưu toàn bộ danh sách sinh viên ra file JSON.
        Format:
        {
          "students": [
            {...}, {...}
          ]
        }
        """
        data = {"students": [s.to_dict() for s in self._list]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """
        Đọc danh sách sinh viên từ file JSON.
        Dùng StudentBase.from_dict để tự động tạo đúng loại sinh viên.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._list = [StudentBase.from_dict(d) for d in data.get("students", [])]


# ========================
# HÀM HỖ TRỢ MENU / NHẬP DỮ LIỆU
# ========================
def input_int(msg, lo=None, hi=None):
    """
    Nhập 1 số nguyên, có kiểm tra khoảng [lo, hi] nếu có.
    Lặp lại cho đến khi nhập đúng.
    """
    while True:
        try:
            v = int(input(msg))
            if lo is not None and v < lo:
                print(f">= {lo}")
                continue
            if hi is not None and v > hi:
                print(f"<= {hi}")
                continue
            return v
        except Exception:
            print("Nhập số hợp lệ!")


def create_student():
    """
    Nhập thông tin 1 sinh viên mới từ bàn phím:
    - ID, tên, năm
    - Loại (ĐH hay CH)
    - (nếu CH) nhập thêm advisor
    - Hỏi có muốn thêm môn không -> nếu có thì nhập nhiều môn
    Trả về 1 object Undergraduate hoặc Graduate.
    """
    sid = input("ID: ").strip()
    name = input("Tên: ").strip()
    year = input_int("Năm (1..6): ", 1, 6)
    t = input("Loại (u=ĐH, g=CH): ").strip().lower()

    if t == "u":
        s = Undergraduate(sid, name, year)
    elif t == "g":
        adv = input("Người hướng dẫn (bỏ trống nếu không): ").strip()
        s = Graduate(sid, name, year, adv)
    else:
        raise ValueError("Loại không hợp lệ")

    # thêm môn tuỳ chọn
    while True:
        more = input("Thêm môn? (y/n): ").strip().lower()
        if more != "y":
            break
        cname = input("  Tên môn: ").strip()
        ccre = input_int("  Tín chỉ: ", 1, None)
        clet = input("  Điểm (A/B/C/D/F): ").strip().upper()
        s.add_course(cname, ccre, clet)
    return s


def show_detail(s: StudentBase):
    """
    In chi tiết 1 sinh viên:
    - ID, tên, loại, advisor (nếu có)
    - Năm, GPA, học phí, trạng thái
    - Danh sách môn học
    """
    if s is None:
        print("Không có sinh viên để hiển thị.")
        return

    print("\n--- CHI TIẾT ---")
    print("ID      :", s.id)
    print("Tên     :", s.name)
    print("Loại    :", s.student_type())
    if isinstance(s, Graduate):
        print("Advisor :", s.advisor or "(không)")
    print("Năm     :", s.year)
    print("GPA     :", s.gpa())
    print("Học phí :", s.tuition(), "VND")
    print("Trạng thái:", s.status())
    print("Môn học :", len(s.courses))
    for i, c in enumerate(s.courses, 1):
        print(f"  {i:02d}. {c['name']} - {c['credits']} tín chỉ - {c['letter']}")


# ========================
# MENU CHÍNH (chương trình console)
# ========================
def main():
    mgr = StudentManager()

    # Demo sẵn 2 sinh viên để khỏi phải nhập tay ngay từ đầu
    u = Undergraduate("SV001", "Nguyễn Văn A", 3)
    u.add_course("CTDL", 4, "B")
    u.add_course("Toán Rời Rạc", 3, "A")
    mgr.add(u)

    g = Graduate("SV100", "Trần Thị B", 1, "GS. Hồ")
    g.add_course("ML Nâng Cao", 3, "A")
    g.add_course("NLP", 3, "A")
    mgr.add(g)

    MENU = """
===== MENU QLSV (đơn giản) =====
1) Thêm sinh viên
2) Xoá sinh viên
3) Thêm môn cho SV
4) Tìm theo ID (in chi tiết)
5) Tìm theo tên (bảng)
6) Liệt kê theo GPA
7) Liệt kê theo TÊN
8) Lưu JSON
9) Tải JSON
0) Thoát
"""

    while True:
        # In menu + hỏi lựa chọn
        print(MENU)
        ch = input("Chọn: ").strip()
        try:
            if ch == "1":
                # Thêm sinh viên mới
                s = create_student()
                mgr.add(s)
                print("OK, đã thêm.")
            elif ch == "2":
                # Xóa sinh viên theo ID
                sid = input("ID: ").strip()
                print("Đã xoá." if mgr.remove(sid) else "Không tìm thấy.")
            elif ch == "3":
                # Thêm môn cho 1 sinh viên
                sid = input("ID: ").strip()
                s = mgr.find_by_id(sid)
                if not s:
                    print("Không thấy.")
                    continue
                cname = input("  Tên môn: ").strip()
                ccre = input_int("  Tín chỉ: ", 1, None)
                clet = input("  Điểm (A/B/C/D/F): ").strip().upper()
                s.add_course(cname, ccre, clet)
                print("Đã thêm môn.")
            elif ch == "4":
                # Tìm theo ID và in chi tiết
                sid = input("ID: ").strip()
                s = mgr.find_by_id(sid)
                show_detail(s) if s else print("Không thấy.")
            elif ch == "5":
                # Tìm theo tên, in bảng tóm tắt
                kw = input("Từ khoá tên: ").strip()
                mgr.print_table(mgr.report_rows(mgr.search_by_name(kw)))
            elif ch == "6":
                # Liệt kê theo GPA giảm dần
                mgr.print_table(mgr.report_rows(mgr.sort_by_gpa(True)))
            elif ch == "7":
                # Liệt kê theo tên a->z
                mgr.print_table(mgr.report_rows(mgr.sort_by_name()))
            elif ch == "8":
                # Lưu JSON
                p = input("File (vd: students.json): ").strip()
                mgr.save(p)
                print("Đã lưu.")
            elif ch == "9":
                # Tải JSON
                p = input("File: ").strip()
                mgr.load(p)
                print("Đã tải.")
            elif ch == "0":
                # Thoát chương trình
                print("Bye!")
                break
            else:
                print("Chọn không hợp lệ.")
        except Exception as e:
            # Bắt mọi lỗi phát sinh trong xử lý menu để tránh crash
            print("Lỗi:", e)


if __name__ == "__main__":
    # Chỉ chạy main nếu file này được chạy trực tiếp
    main()
