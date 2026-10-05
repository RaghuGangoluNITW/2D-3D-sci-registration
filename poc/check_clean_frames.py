import sys, json
sys.path.insert(0, '.')
sys.path.insert(0, r'C:\Users\rgangolu\OneDrive - Infor\analysis\mslevel_reg\src')
import numpy as np, cv2, pydicom
from scipy.optimize import minimize
from similarity import gradient_orientation_similarity
from load_jyothi_ct import load_jyothi
from drr_renderer import CTVolume, DRRRenderer
import run_jyothi_go_test as m

BANK_DIR = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\jyothi_data\drr_bank'
manifest = json.loads(open(BANK_DIR + '/manifest.json').read())
all_entries = manifest['entries']
out_size = manifest['output_size']
pix_mm_bank = manifest['pixel_spacing_mm']
sid_mm = manifest['sid_mm']

data = load_jyothi()
ct = CTVolume(data['volume_zyx'], data['spacing_xyz'], data['origin_xyz'], hu_threshold=180.0)
renderer = DRRRenderer(ct)
f_px = sid_mm / pix_mm_bank
cx = cy = (out_size - 1) / 2.0
K = np.array([[f_px, 0, cx], [0, f_px, cy], [0, 0, 1]])

for fidx, viewtag in [(1, 'LAT'), (2, 'LAT')]:
    path = r'C:\Users\rgangolu\OneDrive - Infor\analysis\jyothi\carm_extracted\jyothi carm\20260707\0\{}'.format(fidx)
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    arr = arr / arr.max()
    target = m.preprocess_xray(arr, out_size)
    entries = [e for e in all_entries if e['base_view'] == viewtag]
    scores = [gradient_orientation_similarity(
        cv2.imread(BANK_DIR + '/' + e['file'], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0,
        target, sigma=2.0) for e in entries]
    best = entries[int(np.argmax(scores))]
    print('frame', fidx, viewtag, 'best retrieval GO=', round(max(scores), 3),
          'az=', round(best['azimuth_deg'], 1), 'el=', round(best['elevation_deg'], 1))
    src0 = np.array(best['src_xyz']); pax0 = np.array(best['pax_xyz'])
    u0 = np.array(best['u_xyz']); v0 = np.array(best['v_xyz'])
    R_pnp = np.stack([u0, v0, pax0], axis=0)
    t_pnp = -R_pnp @ src0

    def pose_to_basis(R, t):
        return -R.T @ t, R[2], R[0], R[1]

    def render_from_Rt(R, t, n_steps=140):
        src, pax, u, v = pose_to_basis(R, t)
        return renderer.render(src, pax, u, v, output_size=out_size,
                                pixel_spacing_mm=pix_mm_bank, n_steps=n_steps, sdd_mm=sid_mm)

    def rt_from_delta(x0, R_pnp=R_pnp, t_pnp=t_pnp):
        drx, dry, drz, dtx, dty, dtz = x0
        Rd = cv2.Rodrigues(np.array([drx, dry, drz]) * np.pi / 180.0)[0]
        return Rd @ R_pnp, t_pnp + np.array([dtx, dty, dtz])

    def go_sim(a, b):
        return gradient_orientation_similarity(a, b, sigma=2.0)

    BOUNDS = np.array([15., 15., 15., 25., 25., 25.])

    def cost(x0):
        if np.any(np.abs(x0) > BOUNDS):
            return 1.0
        R2, t2 = rt_from_delta(x0)
        drr2 = render_from_Rt(R2, t2)
        cov = np.count_nonzero(drr2) / max(drr2.size, 1)
        if cov < 0.1:
            return 1.0
        return 1.0 - go_sim(drr2, target)

    x0 = np.zeros(6)
    step = np.array([3., 3., 3., 5., 5., 5.])
    simplex = np.vstack([x0] + [x0 + step[i] * np.eye(6)[i] for i in range(6)])
    res = minimize(cost, x0=x0, method='Nelder-Mead',
                   options=dict(xatol=0.02, fatol=1e-5, maxiter=400, maxfev=400,
                                initial_simplex=simplex, adaptive=True))
    R2, t2 = rt_from_delta(res.x)
    drr_final = render_from_Rt(R2, t2, n_steps=180)
    print('  final cost', round(res.fun, 4), 'coverage',
          round(np.count_nonzero(drr_final) / drr_final.size, 3))
    tgt8 = (np.clip(target, 0, 1) * 255).astype('uint8')
    fin8 = (np.clip(drr_final, 0, 1) * 255).astype('uint8')
    out_path = r'C:\Users\rgangolu\OneDrive - Infor\analysis\POC\results\clean_frame{}_LAT_result.png'.format(fidx)
    cv2.imwrite(out_path, np.hstack([tgt8, fin8]))
