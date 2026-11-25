# NCSI 3.0 group fulfilment (points / max) and NRI overall score (/100)
# Order: CII, IR, Policy, R&D, Education, Crisis, Cybercrime, Awareness, PersonalData, Military, Cooperation, Enablers
MAX = [12, 14, 15, 4, 10, 9, 16, 12, 4, 6, 6, 12]
PTS = {
    # Kazakhstan, NCSI version 14 Feb 2025 (data used in the original manuscript); NRI 2023 = 50.97
    "KZ-2025": ([12, 11, 12, 2, 10, 4, 11, 12, 4, 4, 4, 6], 0.51, 76.67),
    # Kazakhstan, NCSI version 3 Nov 2026; NRI 2025 = 49.06
    "KZ-2026": ([12, 11, 15, 4, 10, 4, 11, 9, 4, 4, 6, 8], 0.4906, 81.67),
    # intermediate NCSI versions (from the NCSI version selector); NRI 2024 = 50.52 (published 21 Nov 2024), NRI 2025 = 49.06 (4 Feb 2026)
    "KZ-2025-09": ([12, 11, 12, 2, 6, 4, 11, 12, 4, 4, 4, 6], 0.5052, 73.33),
    "KZ-2025-11": ([12, 11, 12, 2, 10, 4, 11, 12, 4, 4, 4, 6], 0.5052, 76.67),
    "KZ-2026-07": ([12, 11, 12, 4, 10, 4, 11, 12, 4, 4, 6, 8], 0.4906, 81.67),
    "KZ-2026-08": ([12, 11, 12, 4, 10, 4, 11, 9, 4, 4, 6, 8], 0.4906, 79.17),
    "USA":     ([9, 14, 12, 4, 6, 9, 16, 9, 4, 6, 6, 6], 0.7913, 84.17),
    "Germany": ([12, 14, 9, 4, 8, 4, 16, 9, 4, 4, 6, 12], 0.7412, 85.00),
    "Japan":   ([12, 12, 12, 4, 10, 5, 16, 6, 4, 4, 6, 8], 0.7022, 82.50),
    "Estonia": ([12, 14, 15, 4, 10, 7, 16, 9, 4, 4, 6, 12], 0.6995, 94.17),
    "Uzbekistan": ([12, 11, 6, 2, 5, 2, 8, 9, 4, 0, 3, 4], 0.4647, 55.00),
    "Kyrgyzstan": ([12, 14, 9, 0, 10, 3, 3, 6, 4, 0, 3, 0], 0.4244, 53.33),
}
def X(name):
    p, nri, ncsi = PTS[name]
    return [round(a / b, 2) for a, b in zip(p, MAX)], nri, ncsi

# KZ-CERT registered information-security incidents, Kazakhstan, January-December (cert.gov.kz; data as of 24.11.2026)
INCIDENTS = {2021: 19380, 2022: 16654, 2023: 34481, 2024: 28013, 2025: 61067}
INCIDENTS_2026_PARTIAL = 66765            # 1 Jan - 24 Nov 2026 (328 days)
INCIDENTS_2026_ANNUALISED = INCIDENTS_2026_PARTIAL * 365 / 328
N_MAX = max(INCIDENTS.values())           # 61067 (2025), complete years only
# NRI overall score of Kazakhstan by edition (Portulans Institute)
NRI_KZ = {2021: 0.5217, 2022: 0.5246, 2023: 0.5097, 2024: 0.5052, 2025: 0.4906}
