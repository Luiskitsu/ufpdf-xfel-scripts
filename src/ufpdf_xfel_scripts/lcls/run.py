"""
lcls_run.py
-----------
A module providing the `Run` class, which loads and reduces LCLS pump-probe
diffraction data from an HDF5 file, storing only the reduced arrays as
attributes (the large raw arrays are discarded after reduction).
"""

import logging
import warnings

import h5py
import numpy as np
from diffpy.morph.morphpy import morph_arrays
from diffpy.pdfgetx.pdfconfig import PDFConfig
from diffpy.pdfgetx.pdfgetter import PDFGetter
from diffpy.utils.parsers import load_data

# from src.ufpdf_xfel_scripts.lcls.scripts.morph_LCLS import squeeze
from ufpdf_xfel_scripts.lcls.paths import (
    experiment_data_dir,
    synchrotron_data_dir,
)

warnings.filterwarnings("ignore")
logging.getLogger("diffpy.pdfgetx").setLevel(logging.ERROR)
logging.getLogger("diffpy.pdfgetx.user").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


class Run:
    """Loads and reduces a single LCLS pump-probe run.

    Parameters
    ----------
    run_number : int
        The run number to load (e.g. 22).
    background_number : int
        The background run number (e.g. 1).
    sample_name : str
        The short sample label used in file names (e.g. 'NSPSe').
    sample_composition : str
        The chemical composition string for PDFGetter (e.g. 'Na11SnPSe12').
    instrument : str
        The instrument prefix (e.g. 'mfx').
    experiment_number : str
        The experiment identifier (e.g. 'l1044925').
    target_id : int
        The index of the delay to use as the morph target (default 0).
    q_min : float
        The lower Q bound for the I(Q) figure of merit (default 9).
    q_max : float
        the upper Q bound for the I(Q) figure of merit (default 9.5).
    r_min_fom : float
        The lower r bound for the G(r) figure of merit (default 2).
    r_max_fom : float
        The upper r bound for the G(r) figure of merit (default 5).
    q_min_morph : float
        the lower Q bound for morph normalisation (default 0).
    q_max_morph : float
        The upper Q bound for morph normalisation (default 12).
    scale : float
        The initial scale parameter for morphing (default 1.01).
    stretch : float or None
        The initial stretch parameter for morphing (default None).
    smear : float or None
        The initial smear parameter for morphing (default None).
    points_away_t0_plot_on_off : int
        The number of delay points away from t0 to select for on/off
        plots (default 0).
    verbose : bool
        The verbosity for debugging and assessing (default, False, is
        low verbosity).
    azimuthal_selector : str
        The selection between vertical, horitzontal or total azimuthal
        integration.
    i0_percentile_threshold : float or None
        The percentile of i0 intensities to drop. A value of 5 will filter
        and drop the lowest 5% of i0 intensities.
        (default 5, None disables the filter.)
    jitter_threshold_fs : float
        The threshold for jitter time-offsets to ignore. time-offsets >=
        jitter_threshold will be filtered and dropped. Units are
        femtoseconds. (default 250, None disables the filter.)

    Attributes
    ----------
    q : np.ndarray
        Q-grid (1-D, shape (n_q,)).
    delays : np.ndarray
        Sorted unique delay times in ps (1-D, shape (n_delays,)).
    Is_on : np.ndarray
        Delay-averaged, sorted pump-ON I(Q) (shape (n_delays, n_q)).
    Is_off : np.ndarray
        Delay-averaged, sorted pump-OFF I(Q) (shape (n_delays, n_q)).
    target_delay : float
        The delay value used as the morph target.
    raw_delays : dict
        Dict keyed by delay time containing raw [q, on, off, diff, ...] lists.
    morph_delays : dict
        Dict keyed by delay time containing morphed [q, on, off,
        diff, ...] lists.
    delay_scan : bool
        True if the run contains a delay scan, False otherwise.
    q_synchrotron : np.ndarray
        Q-grid from the synchrotron reference file.
    fq_synchrotron : np.ndarray
        F(Q) from the synchrotron reference file.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        run_number,
        background_number,
        sample_name,
        sample_composition,
        instrument,
        experiment_number,
        number_of_static_samples=11,
        target_id=0,
        q_min=9,
        q_max=9.5,
        r_min_fom=2,
        r_max_fom=5,
        q_min_morph=0,
        q_max_morph=12,
        delay_scale=1.01,
        delay_hshift=None,
        delay_vshift=None,
        delay_stretch=None,
        delay_smear=None,
        points_away_t0_plot_on_off=0,
        verbose=False,
        delay_motor="mfx_lxt_fast2",
        pdfgetter_config=None,
        getx_scale=1,
        getx_squeeze_parms=None,
        fit_qmin=0,
        fit_qmax=12,
        pdf_rmin=0,
        pdf_rmax=60,
        azimuthal_selector=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        i0_percentile_threshold=5,
        jitter_threshold_fs=250.0,
    ):
        # --- store run-level metadata ---
        self.run_number = run_number
        self.background_number = background_number
        self.sample_name = sample_name
        self.sample_composition = sample_composition
        self.instrument = instrument
        self.experiment_number = experiment_number
        self.number_of_static_samples = number_of_static_samples
        self.delay_motor = delay_motor
        self.pdfgetter_config = pdfgetter_config
        self.getx_scale = getx_scale
        self.getx_squeeze = getx_squeeze_parms
        self.verbose = verbose
        self.bad_background_bool = False
        self.azimuthal_selector = azimuthal_selector

        # --- store setup parameters ---
        self.target_id = target_id
        self.q_min = q_min
        self.q_max = q_max
        self.pdf_rmin = pdf_rmin
        self.pdf_rmax = pdf_rmax
        self.fit_qmin = fit_qmin
        self.fit_qmax = fit_qmax
        self.r_min_fom = r_min_fom
        self.r_max_fom = r_max_fom
        self.morph_params = {
            "xmin": q_min_morph,
            "xmax": q_max_morph,
            "scale": delay_scale,
            "hshift": delay_hshift,
            "vshift": delay_vshift,
            "stretch": delay_stretch,
            "smear": delay_smear,
        }
        self.points_away_t0_plot_on_off = points_away_t0_plot_on_off

        # --- filtering defaults ---
        self.i0_percentile_threshold = i0_percentile_threshold
        self.jitter_threshold_fs = jitter_threshold_fs
        self.plt_filter_cutoff_diode = None
        # Convert jitter_threshold from femtoseconds to picoseconds
        # if not 'None' type (1 ps = 1e3 fs).
        self.plt_filter_cutoff_time = (
            None
            if self.jitter_threshold_fs is None
            else float(self.jitter_threshold_fs) * 1e-3
        )

        # --- run the reduction pipeline ---
        self._load()
        self._filter()
        self._reduce()
        self._morph()
        try:
            self._transform()
        except SystemExit:
            print(
                "WARNING: the refinement failed, please re-run using other "
                "fit_qmax and getter_config"
            )
        self._cleanup()

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_evenly(intensities, monitor1, monitor2, n_points):
        N = len(intensities)
        indices = np.linspace(0, N - 1, n_points, dtype=int)
        unnormalized = intensities[indices]
        monitor1_normalized = intensities[indices] / monitor1[indices]
        monitor2_normalized = intensities[indices] / monitor2[indices]
        return unnormalized, monitor1_normalized, monitor2_normalized

    def compute_gr(self, q, fq):
        """FT by hand to preserve the morph squeezes.

        Can't use the raw output from pdfGetter for this reason.

        Parameters
        ----------
        q
        fq
        r_min
        r_max
        npoints

        Returns
        -------
        """
        r = np.arange(self.pdf_rmin, self.pdf_rmax, 0.01)
        qr = np.outer(q, r)
        integrand = fq[:, None] * np.sin(qr)
        gr = (2 / np.pi) * np.trapezoid(integrand, q, axis=0)
        return r, gr

    def _average_equal_times(self):
        # average repeated delays
        self.unique_delays = np.unique(self.delays)
        Is_avg_on = []
        Is_avg_off = []
        Is_avg_on_mon1 = []
        Is_avg_off_mon1 = []
        Is_avg_on_mon2 = []
        Is_avg_off_mon2 = []
        valid_delays = []

        for unique_delay in self.unique_delays:
            mask_on = (self.delays == unique_delay) & self.laser_mask
            mask_off = (self.delays == unique_delay) & ~self.laser_mask
            if not (mask_on.any() and mask_off.any()):
                continue
            valid_delays.append(unique_delay)
            Is_avg_on.append(np.nanmean(self._Is_raw[mask_on], axis=0))
            Is_avg_off.append(np.nanmean(self._Is_raw[mask_off], axis=0))
            Is_avg_on_mon1.append(
                np.nanmean(
                    self._Is_raw[mask_on]
                    / self.monitor1[mask_on].reshape(-1, 1),
                    axis=0,
                )
            )
            Is_avg_off_mon1.append(
                np.nanmean(
                    self._Is_raw[mask_off]
                    / self.monitor1[mask_off].reshape(-1, 1),
                    axis=0,
                )
            )
            Is_avg_on_mon2.append(
                np.nanmean(
                    self._Is_raw[mask_on]
                    / self.monitor2[mask_on].reshape(-1, 1),
                    axis=0,
                )
            )
            Is_avg_off_mon2.append(
                np.nanmean(
                    self._Is_raw[mask_off]
                    / self.monitor2[mask_off].reshape(-1, 1),
                    axis=0,
                )
            )

        self.unique_delays = np.asarray(
            valid_delays
        )  # filter down to valid delay pairs
        self.raw_delays = {}
        for i, step in enumerate(self.unique_delays):
            self.raw_delay_scans = self._build_delay_dict(
                self.raw_delays,
                step,
                self.q,
                Is_avg_on[i],
                Is_avg_off[i],
            )
        self.mon1_normalized_delay_scans = {}
        for i, step in enumerate(self.unique_delays):
            self.mon1_normalized_delay_scans = self._build_delay_dict(
                self.mon1_normalized_delay_scans,
                step,
                self.q,
                Is_avg_on_mon1[i],
                Is_avg_off_mon1[i],
            )
        self.mon2_normalized_delay_scans = {}
        for i, step in enumerate(self.unique_delays):
            self.mon2_normalized_delay_scans = self._build_delay_dict(
                self.mon2_normalized_delay_scans,
                step,
                self.q,
                Is_avg_on_mon2[i],
                Is_avg_off_mon2[i],
            )

    def _morph_fq(self):
        reference_delay = self.delays[self.target_id]

        x_morph = self.morphed_delay_scans[reference_delay][0]
        y_morph = self.morphed_delay_scans[reference_delay][2]
        x_target = self.q_synchrotron
        y_target = self.fq_synchrotron

        morph_table = np.column_stack([x_morph, y_morph])
        target_table = np.column_stack([x_target, y_target])

        self.morphed_fq_parameters, self.morphed_fq = morph_arrays(
            morph_table,
            target_table,
            funcxy=(self.pdfgetter_function, self.pdfgetter_config),
            scale=self.getx_scale,
            squeeze=self.getx_squeeze,
            xmin=self.fit_qmin,
            xmax=self.fit_qmax,
        )
        return

    def _build_delay_dict(self, delay_dict, delay_time, q, on, off):
        """Append one delay entry (mirrors the notebook's
        build_delay_dict)."""
        diff = on - off
        q_min, q_max = self.q_min, self.q_max
        qmin_idx = find_nearest(q, q_min) if q_min is not None else 0
        qmax_idx = find_nearest(q, q_max) if q_max is not None else -1

        l1_diff = np.sum(np.abs(diff[qmin_idx:qmax_idx]))
        l2_diff = np.sum(diff[qmin_idx:qmax_idx] ** 2)
        diff_int = np.sum(diff[qmin_idx:qmax_idx])
        i_sum_off = np.sum(off[qmin_idx:qmax_idx])
        i_sum_on = np.sum(on[qmin_idx:qmax_idx])

        delay_dict[delay_time] = [
            q,
            on,
            off,
            diff,
            l1_diff,
            i_sum_off,
            i_sum_on,
            l2_diff,
            diff_int,
        ]
        return delay_dict

    def _build_parameters_dict(
        self,
        parameter_dict,
        delay_time,
        morph_parameters_on,
        morph_parameters_off,
    ):
        """Append one delay entry (mirrors the notebook's
        build_delay_dict)."""
        parameter_dict[delay_time] = [
            morph_parameters_on,
            morph_parameters_off,
        ]
        return parameter_dict

    def _build_fq_delay_dict(self, fq_delay_dict, delay_t, q, fq_on, fq_off):
        diff = fq_on - fq_off
        q_min, q_max = self.q_min, self.q_max
        qmin_idx = find_nearest(q, q_min) if q_min is not None else 0
        qmax_idx = find_nearest(q, q_max) if q_max is not None else -1

        fq_delay_dict[delay_t] = {
            "q": q,
            "fq_on": fq_on,
            "fq_off": fq_off,
            "diff_gr": diff,
            "RMS": np.sqrt(np.sum(diff[qmin_idx:qmax_idx] ** 2)),
            "diff_int": np.sum(diff[qmin_idx:qmax_idx]),
            "sum_fq_on": np.sum(fq_on[qmin_idx:qmax_idx]),
            "sum_fq_off": np.sum(fq_off[qmin_idx:qmax_idx]),
        }
        return fq_delay_dict

    def _build_gr_delay_dict(self, gr_delay_dict, delay_t, r, gr_on, gr_off):
        diff = gr_on - gr_off
        r_min, r_max = self.pdf_rmin, self.pdf_rmax
        rmin_idx = find_nearest(r, r_min) if r_min is not None else 0
        rmax_idx = find_nearest(r, r_max) if r_max is not None else -1

        gr_delay_dict[delay_t] = {
            "r": r,
            "gr_on": gr_on,
            "gr_off": gr_off,
            "diff_gr": diff,
            "RMS": np.sqrt(np.sum(diff[rmin_idx:rmax_idx] ** 2)),
            "diff_int": np.sum(diff[rmin_idx:rmax_idx]),
            "sum_gr_on": np.sum(gr_on[rmin_idx:rmax_idx]),
            "sum_gr_off": np.sum(gr_off[rmin_idx:rmax_idx]),
        }
        return gr_delay_dict

    def _cleanup(self):
        del self._Is_raw

    def pdfgetter_function(self, q, y, **kwargs):
        """Transform input I(Q) to F(Q) using PDFGetter with background
        correction.

        Returns (x, F(Q)).
        """
        cfg = PDFConfig()
        cfg.composition = self.sample_composition
        cfg.rpoly = self.pdfgetter_config.get("rpoly", 0.9)
        cfg.qmin = self.pdfgetter_config.get("qmin", 0.5)
        cfg.qmax = self.pdfgetter_config.get("qmax", 15.0)
        cfg.qmaxinst = self.pdfgetter_config.get("qmaxinst", 15.0)
        cfg.dataformat = "QA"
        cfg.mode = "xray"
        cfg.bgscale = self.pdfgetter_config.get("bgscale")
        if cfg.bgscale[0] != 0:
            background_path = experiment_data_dir
            if background_path.isfile():
                bg_data = load_data(background_path)
                if bg_data.ndim == 2 and bg_data.shape[1] == 2:
                    bg_interp = np.interp(q, bg_data[:, 0], bg_data[:, 1])
                else:
                    bg_interp = bg_data  # already on same grid
                cfg.background = bg_interp
                cfg.bgscale = cfg.bgscale
        else:
            if not self.bad_background_bool:
                print(
                    """WARNING: no background subtraction carried out,
                set bgscale in pdfgetter_config to a non zero value to carry
                out background subtraction"""
                )
                self.bad_background_bool = True
            cfg.background = None

        # Run PDFGetter
        pg = PDFGetter(cfg)
        _, _ = pg(x=q, y=y)
        q_out, fq_out = pg.fq

        fq_interp = np.interp(q, q_out, fq_out)
        return q, fq_interp

    # ------------------------------------------------------------------
    # pipeline steps
    # ------------------------------------------------------------------

    def _load(self):
        """Load the HDF5 file and synchrotron reference; store only
        reduced arrays."""
        str_run_number = str(self.run_number).zfill(4)
        h5_filename = (
            f"{self.instrument}{self.experiment_number}_Run{str_run_number}.h5"
        )
        input_path = experiment_data_dir / h5_filename
        synchrotron_path = synchrotron_data_dir / f"{self.sample_name}_room.fq"

        # synchrotron reference
        self.q_synchrotron, self.fq_synchrotron = load_data(
            synchrotron_path, unpack=True
        )

        if not input_path.is_file():
            raise FileNotFoundError(f"HDF5 file not found: {input_path}")

        with h5py.File(input_path, "r") as f:
            qs = np.asarray(f["jungfrau"]["pyfai_q"][:])
            Is_raw = np.asarray(f["jungfrau"]["pyfai_azav"][:])
            Is_raw = np.nanmean(Is_raw[:, self.azimuthal_selector, :], axis=1)
            monitor1 = np.asarray(f["MfxDg1BmMon/totalIntensityJoules"][:])
            monitor2 = np.asarray(f["MfxDg2BmMon/totalIntensityJoules"][:])
            time_jitter = np.asarray(f["/tt/fltpos_ps"][:])
            timestamp = np.asarray(f["/timestamp"][:])

            self.delay_scan = (
                "scan" in f
            )  # true if scan exists in the dataset, false otherwise
            if self.delay_scan:
                delays = f["scan"][self.delay_motor][:].squeeze() * 1e12
                self.target_delay = delays[self.target_id]
            else:
                delays = None  # filled below
                self.target_delay = None

            laser_mask = f["lightStatus"]["laser"][:].astype(bool)
            xray_mask = f["lightStatus"]["xray"][:].astype(bool)

        if self.verbose:
            print("shape of qs:", qs.shape)
            print("shape of Is_raw:", Is_raw.shape)
            print("delay_scan:", self.delay_scan)

        # separate x-ray darks and lights
        self.q = qs[0]
        self._Is_raw = Is_raw[xray_mask].copy()
        self.monitor1 = monitor1[xray_mask].copy()
        self.monitor2 = monitor2[xray_mask].copy()
        self.darks = Is_raw[~xray_mask].copy()
        self.delays = delays[xray_mask].copy()
        self.laser_mask = laser_mask[xray_mask].copy()
        self.time_jitter = time_jitter[xray_mask].copy()
        self.timestamp = timestamp[xray_mask].copy()
        return

    def _filter(self):
        """Filter shots using monitor2 (diode) and time_jitter
        metadata."""
        # filter 1: monitor2 (diode)
        if self.i0_percentile_threshold is not None:
            self.plt_filter_pre_monitor2 = self.monitor2.copy()
            if self.timestamp is not None:  # timestamp used in filter 1
                self.plt_filter_pre_timestamp = self.timestamp.copy()
            if not (0.0 < float(self.i0_percentile_threshold) < 100.0):
                raise ValueError(
                    "i0_percentile_threshold must be in (0, 100)."
                )
            keep_diode_mask = np.isfinite(self.monitor2) & (
                self.monitor2 != 0.0
            )
            if keep_diode_mask.any():
                self.plt_filter_cutoff_diode = float(
                    np.quantile(
                        self.monitor2[keep_diode_mask],
                        self.i0_percentile_threshold / 100.0,
                    )
                )
                keep_diode_mask &= (
                    self.monitor2 >= self.plt_filter_cutoff_diode
                )
            self.plt_filter_keep_diode = keep_diode_mask.copy()
            if self.verbose:
                removed = int(np.sum(~keep_diode_mask))
                total = int(self.monitor2.shape[0])
                zeros = int(
                    np.sum(np.isfinite(self.monitor2) & (self.monitor2 == 0.0))
                )
                msg = (
                    "[DBG] _filter diode (mon2):\n"
                    f"  removed: {removed}/{total}\n"
                    f"  zeros: {zeros}\n"
                    f"  i0 pct threshold: {self.i0_percentile_threshold}\n"
                    f"  low quantile cutoff: {self.plt_filter_cutoff_diode}"
                )
                print(msg)
            self._Is_raw = self._Is_raw[keep_diode_mask, :].copy()
            self.monitor1 = self.monitor1[keep_diode_mask].copy()
            self.monitor2 = self.monitor2[keep_diode_mask].copy()
            self.laser_mask = self.laser_mask[keep_diode_mask].copy()
            if self.time_jitter is not None:
                self.time_jitter = self.time_jitter[keep_diode_mask].copy()
            if bool(self.delay_scan) and self.delays is not None:
                self.delays = self.delays[keep_diode_mask].copy()
            if self.timestamp is not None:
                self.timestamp = self.timestamp[keep_diode_mask].copy()

        # filter 2: time_jitter
        if self.jitter_threshold_fs is None:
            return
        finite_time_mask = np.isfinite(self.time_jitter)
        if not finite_time_mask.any():
            return
        if float(self.jitter_threshold_fs) <= 0.0:
            raise ValueError("jitter_threshold_fs must be > 0 or None.")
        self.plt_filter_pre_time = self.time_jitter.copy()
        jitter_negative_count = int(
            np.sum(self.time_jitter[finite_time_mask] < 0.0)
        )
        keep_time_mask = finite_time_mask & (self.time_jitter >= 0.0)
        keep_time_mask &= self.time_jitter <= self.plt_filter_cutoff_time
        if hasattr(self, "plt_filter_keep_diode"):
            self.plt_filter_keep_time = np.zeros(
                self.plt_filter_keep_diode.shape, dtype=bool
            )
            self.plt_filter_keep_time[self.plt_filter_keep_diode] = (
                keep_time_mask
            )
        else:
            self.plt_filter_keep_time = keep_time_mask.copy()
        if self.verbose:
            min_ps = float(np.min(self.time_jitter[finite_time_mask]))
            max_ps = float(np.max(self.time_jitter[finite_time_mask]))
            msg = (
                "[DBG] _filter time jitter:\n"
                f"  threshold (fs): {self.jitter_threshold_fs}\n"
                f"  threshold (ps): {self.plt_filter_cutoff_time}\n"
                f"  negative time jitter count: {jitter_negative_count}\n"
                f"  removed: {int(np.sum(~keep_time_mask))}"
                f"/{int(self.time_jitter.shape[0])}\n"
                f"  min_ps: {min_ps:.3f}\n"
                f"  max_ps: {max_ps:.3f}"
            )
            print(msg)
        self._Is_raw = self._Is_raw[keep_time_mask, :].copy()
        self.monitor1 = self.monitor1[keep_time_mask].copy()
        self.monitor2 = self.monitor2[keep_time_mask].copy()
        self.time_jitter = self.time_jitter[keep_time_mask].copy()
        self.laser_mask = self.laser_mask[keep_time_mask].copy()
        if bool(self.delay_scan) and self.delays is not None:
            self.delays = self.delays[keep_time_mask].copy()
        if self.timestamp is not None:
            self.timestamp = self.timestamp[keep_time_mask].copy()

    def _reduce(self):
        """Build raw_delays dict (unmorphed) from the reduced arrays."""
        if not self.delay_scan:
            (
                self.subsample,
                self.subsample_monitor1,
                self.subsample_monitor2,
            ) = self._sample_evenly(self.delays, self.number_of_static_samples)
        else:
            self._average_equal_times()

    def _morph(self):
        """Apply diffpy.morph to each delay and store results in
        morph_delays."""
        params = self.morph_params
        target = self.raw_delays[self.target_delay]
        target_table = np.column_stack([target[0], target[1]])

        self.morph_parameters = {}
        self.morphed_delay_scans = {}
        for delay_t, data in self.raw_delays.items():
            x = data[0]
            y_on = data[1]
            y_off = data[2]

            morph_on_table = np.column_stack([x, y_on])
            morph_off_table = np.column_stack([x, y_off])

            # fit morph parameters
            morph_parameters_on, _ = morph_arrays(
                morph_on_table, target_table, **params
            )
            morph_parameters_off, _ = morph_arrays(
                morph_off_table, target_table, **params
            )

            # apply parameters without refining.  This is a workaround
            # because of limited range of x that morph returns
            _, table_on_full = morph_arrays(
                morph_on_table,
                target_table,
                scale=morph_parameters_on.get("scale"),
                hshift=morph_parameters_on.get("hshift"),
                vshift=morph_parameters_on.get("vshift"),
                stretch=morph_parameters_on.get("stretch"),
                smear=morph_parameters_on.get("smear"),
                apply=True,
            )
            _, table_off_full = morph_arrays(
                morph_off_table,
                target_table,
                scale=morph_parameters_off.get("scale"),
                hshift=morph_parameters_off.get("hshift"),
                vshift=morph_parameters_off.get("vshift"),
                stretch=morph_parameters_off.get("stretch"),
                smear=morph_parameters_off.get("smear"),
                apply=True,
            )

            on_morph = table_on_full[:, 1]
            off_morph = table_off_full[:, 1]

            self.morphed_delay_scans = self._build_delay_dict(
                self.morphed_delay_scans, delay_t, x, on_morph, off_morph
            )

            self.morph_parameters = self._build_parameters_dict(
                self.morph_parameters,
                delay_t,
                morph_parameters_on,
                morph_parameters_off,
            )

    def _transform(self):
        """Apply diffpy.pdfgetx to each delay and store results in
        morph_delays."""
        self.fq_delay_scans = {}
        self.gr_delay_scans = {}
        self._morph_fq()

        for delay_t in self.morphed_delay_scans:
            q_iq = self.morphed_delay_scans[delay_t][0]
            on_iq = self.morphed_delay_scans[delay_t][1]
            off_iq = self.morphed_delay_scans[delay_t][2]

            table_on = np.column_stack([q_iq, on_iq])
            table_off = np.column_stack([q_iq, off_iq])

            target_dummy = table_on  # only to preserve grid

            _, fq_on = morph_arrays(
                table_on,
                target_dummy,
                funcxy=(
                    self.pdfgetter_function,
                    self.morphed_fq_parameters["funcxy"],
                ),
                scale=float(self.morphed_fq_parameters["scale"]),
                squeeze=[
                    float(v)
                    for v in self.morphed_fq_parameters["squeeze"].values()
                ],
                xmin=self.fit_qmin,
                xmax=self.fit_qmax,
                apply=True,
            )

            _, fq_off = morph_arrays(
                table_off,
                target_dummy,
                funcxy=(
                    self.pdfgetter_function,
                    self.morphed_fq_parameters["funcxy"],
                ),
                scale=float(self.morphed_fq_parameters["scale"]),
                squeeze=[
                    float(v)
                    for v in self.morphed_fq_parameters["squeeze"].values()
                ],
                xmin=self.fit_qmin,
                xmax=self.fit_qmax,
                apply=True,
            )

            q_final = fq_on[:, 0]
            fq_on = fq_on[:, 1]
            fq_off = fq_off[:, 1]
            self.fq_delay_scans = self._build_fq_delay_dict(
                self.fq_delay_scans, delay_t, q_final, fq_on, fq_off
            )

            r, gr_on = self.compute_gr(q_final, fq_on)
            r, gr_off = self.compute_gr(q_final, fq_off)
            self.gr_delay_scans = self._build_gr_delay_dict(
                self.gr_delay_scans, delay_t, r, gr_on, gr_off
            )
        return
