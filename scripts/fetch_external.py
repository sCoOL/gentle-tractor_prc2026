# Download the open external sources into PRC_EXTERNAL (default external/). Safe to re-run: finished files are skipped,
# partial downloads are never kept (temp file + rename), each source is isolated, METAR falls back
# from monthly to 7-day windows when the IEM server resets the connection (it does for EHAM).
import os, io, json, time, hashlib, urllib.request, base64, zipfile, glob
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prc_config  # noqa: E402

ROOT = str(prc_config.EXTERNAL); os.makedirs(ROOT, exist_ok=True)   # PRC_EXTERNAL
UA = {'User-Agent': 'Mozilla/5.0 (research; PRC2026 data challenge)'}
LOG = []; FAILED = []
now = lambda: pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M')


def fetch(url, path, timeout=600, tries=4, quiet=False):
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True); tmp = path + '.part'
    for k in range(1, tries + 1):
        t = time.time(); n = 0
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r, open(tmp, 'wb') as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk); n += len(chunk)
            os.replace(tmp, path)
            LOG.append({'file': path, 'url': url, 'bytes': n, 'sha256_16': hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16], 'fetched_utc': now(), 'seconds': round(time.time() - t, 1)})
            if not quiet:
                print(f'  {path} {n/1e6:.1f} MB {time.time()-t:.0f}s', flush=True)
            return True
        except Exception as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            if not quiet:
                print(f'  retry {k}/{tries} {os.path.basename(path)}: {e!r}'[:140], flush=True)
            time.sleep(3 * k)
    FAILED.append(url); return False


APTS = ['LTFM', 'EHAM', 'EGLL', 'LFPG', 'EDDF', 'LEMD', 'LEBL', 'EDDM', 'LIRF', 'LSZH']
FIELDS = 'tmpf,dwpf,relh,drct,sknt,p01i,alti,vsby,gust,skyc1,skyc2,skyc3,skyl1,skyl2,skyl3,wxcodes,metar'


def iem_url(st, a, b):
    return (f'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={st}&data={FIELDS}'
            f'&year1={a.year}&month1={a.month}&day1={a.day}&year2={b.year}&month2={b.month}&day2={b.day}'
            f'&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3')


# ---------------- 1. METAR ----------------
try:
    print('1. METAR from Iowa Environmental Mesonet (monthly, fallback weekly)', flush=True)
    for st in APTS:
        got = 0; weekly = 0
        for pm in pd.period_range('2025-01', '2026-07', freq='M'):
            a = pm.start_time.normalize(); b = (pm + 1).start_time.normalize()
            p = f'{ROOT}/metar/raw/{st}_{pm}.csv'
            if fetch(iem_url(st, a, b), p, timeout=300, tries=2, quiet=True):
                got += 1; continue
            c = a
            while c < b:
                d_ = min(c + pd.Timedelta(days=7), b)
                pw = f'{ROOT}/metar/raw/{st}_{c.date()}_{d_.date()}.csv'
                if fetch(iem_url(st, c, d_), pw, timeout=300, tries=3, quiet=True):
                    weekly += 1
                c = d_; time.sleep(1)
            time.sleep(1)
        print(f'  {st}: {got} monthly files, {weekly} weekly files', flush=True)
    frames = []
    for f in sorted(glob.glob(f'{ROOT}/metar/raw/*.csv')):
        df = pd.read_csv(f, na_values=['M'], low_memory=False)
        if len(df):
            df['valid'] = pd.to_datetime(df['valid'], utc=True); frames.append(df)
    m = pd.concat(frames, ignore_index=True).drop_duplicates(['station', 'valid']).sort_values(['station', 'valid'])
    m.to_parquet(f'{ROOT}/metar/metar_all.parquet')
    cov = m.groupby('station').valid.agg(['min', 'max', 'size'])
    print('  metar_all.parquet rows', len(m)); print(cov.to_string(), flush=True)
except Exception as e:
    print('SECTION 1 (METAR) FAILED', repr(e)[:200], flush=True)

# ---------------- 2. OurAirports ----------------
try:
    print('2. OurAirports', flush=True)
    for f in ['airports.csv', 'runways.csv']:
        fetch(f'https://davidmegginson.github.io/ourairports-data/{f}', f'{ROOT}/ourairports/{f}')
    rw = pd.read_csv(f'{ROOT}/ourairports/runways.csv'); rw[rw.airport_ident.isin(APTS)].to_csv(f'{ROOT}/ourairports/runways_10apts.csv', index=False)
    ap = pd.read_csv(f'{ROOT}/ourairports/airports.csv'); ap[ap.ident.isin(APTS)].to_csv(f'{ROOT}/ourairports/airports_10apts.csv', index=False)
    print('  runways for the 10 airports:', int(rw.airport_ident.isin(APTS).sum()), flush=True)
except Exception as e:
    print('SECTION 2 (OurAirports) FAILED', repr(e)[:200], flush=True)

# ---------------- 3. X-Plane Gateway ----------------
try:
    print('3. X-Plane Scenery Gateway apt.dat', flush=True)
    mp = f'{ROOT}/xplane/manifest.json'; man = json.load(open(mp)) if os.path.exists(mp) else {}
    for icao in APTS:
        p = f'{ROOT}/xplane/{icao}.dat'
        if os.path.exists(p) and icao in man and 'error' not in man[icao]:
            continue
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(f'https://gateway.x-plane.com/apiv1/airport/{icao}', headers=UA), timeout=120).read())
            sid = j['airport']['recommendedSceneryId']; sc = [s for s in j['airport']['scenery'] if s['sceneryId'] == sid]
            s = json.loads(urllib.request.urlopen(urllib.request.Request(f'https://gateway.x-plane.com/apiv1/scenery/{sid}', headers=UA), timeout=300).read())
            z = zipfile.ZipFile(io.BytesIO(base64.b64decode(s['scenery']['masterZipBlob'])))
            name = [n for n in z.namelist() if n.lower().endswith('.dat')][0]
            os.makedirs(f'{ROOT}/xplane', exist_ok=True); open(p, 'wb').write(z.read(name))
            lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
            info = {'sceneryId': sid, 'file_in_pack': name, 'pack_meta': sc[0] if sc else None,
                    'n_startup_locations': sum(1 for l in lines if l.startswith('1300 ')),
                    'n_taxi_nodes': sum(1 for l in lines if l.startswith('1201 ')),
                    'n_taxi_edges': sum(1 for l in lines if l.startswith('1202 ')),
                    'n_runways': sum(1 for l in lines if l.startswith('100 ')), 'fetched_utc': now()}
            man[icao] = info
            print(f"  {icao}: scenery {sid}, stands {info['n_startup_locations']}, nodes {info['n_taxi_nodes']}, edges {info['n_taxi_edges']}, runways {info['n_runways']}", flush=True)
            time.sleep(1)
        except Exception as e:
            print(f'  {icao} FAIL {e!r}'[:220], flush=True); man[icao] = {'error': repr(e)[:200]}
        json.dump(man, open(mp, 'w'), indent=1, default=str)
except Exception as e:
    print('SECTION 3 (X-Plane) FAILED', repr(e)[:200], flush=True)

# ---------------- 4. EUROCONTROL ----------------
try:
    print('4. EUROCONTROL performance data (ansperformance.eu/data)', flush=True)
    for f in ['Taxi-Out_Additional_Time.xlsx', 'ASMA_Additional_Time.xlsx', 'Airport_Arrival_ATFM_Delay.xlsx', 'Airport_Traffic.xlsx']:
        fetch(f'https://www.eurocontrol.int/performance/data/download/xls/{f}', f'{ROOT}/eurocontrol/{f}', timeout=1800)
except Exception as e:
    print('SECTION 4 (EUROCONTROL) FAILED', repr(e)[:200], flush=True)

# ---------------- 5. EUROCONTROL daily pre-departure delay (post-steps, tools/last_day/) ----------------
# All pre-departure delay per airport and day (DLY_ALL_PRE_2, minutes; FLT_DEP_IFR_2 departures). The files grow as days
# are published and can be revised; the final file was built with the copies whose sha256 is listed here.
EC_DAILY_SHA256 = {
    2025: 'd742eeb626e5f9f7e0c756e2ae3cf90078f4627e989a2e31b8ed21ba8f5510be',
    2026: '5952d20ef5f9c4245018dd6986064c5c4aa15d568a6c0c2dcdcbc73bf8a43fd2',
}
try:
    print('5. EUROCONTROL daily all pre-departure delay (ansperformance.eu/data)', flush=True)
    for year, want in EC_DAILY_SHA256.items():
        p = f'{ROOT}/eurocontrol/all_pre_departure_delays_{year}.csv'
        fetch(f'https://www.eurocontrol.int/performance/data/download/csv/all_pre_departure_delays_{year}.csv', p, timeout=600)
        if os.path.exists(p):
            got = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            print(f'  {os.path.basename(p)} sha256 {got}' + ('' if got == want else f' (the final file used {want}; the source was revised)'), flush=True)
except Exception as e:
    print('SECTION 5 (EUROCONTROL daily) FAILED', repr(e)[:200], flush=True)

lp = f'{ROOT}/_fetch_log.csv'
if LOG:
    pd.DataFrame(LOG).to_csv(lp, index=False, mode='a', header=not os.path.exists(lp))
print('ALL DONE; failed urls:', len(FAILED), flush=True)
for u in FAILED[:20]:
    print('  FAILED', u[:140])
