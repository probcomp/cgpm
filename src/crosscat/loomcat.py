# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016 MIT Probabilistic Computing Project

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv
import itertools
import os

import loom.cleanse
import loom.tasks
import pandas as pd

from distributions.io.stream import json_load
from distributions.io.stream import open_compressed
from loom.cFormat import assignment_stream_load

from cgpm.mixtures.view import View
from cgpm.utils import config as cu
from cgpm.utils.parallel_map import parallel_map


DEFAULT_LOOM_STORE = os.path.join(os.sep, 'tmp', 'cgpm', 'loomstore')

DEFAULT_CLEANSED_DIR = 'cleansed'
DEFAULT_RAW_DIR = 'raw'
DEFAULT_RESULTS_DIR = 'results'


def _generate_column_names(state):
    """Returns list of dummy names for the outputs of `state`."""
    return [unicode('c%05d') % (i,) for i in state.outputs]


def _generate_loom_stattypes(state):
    """Returns list of loom stattypes from the cgpm stattypes of `state`."""
    cctypes = state.cctypes()
    distargs = state.distargs()
    return [cu.loom_stattype(s, d) for s, d in zip(cctypes, distargs)]


def _generate_project_paths(name=None):
    """Creates a new project in the loom store."""
    if name is None:
        name = cu.timestamp()
    store = _retrieve_loom_store()
    project_root = os.path.join(store, name)
    # Create necessary subdirectories.
    paths = {
        'root'      : project_root,
        'raw'       : os.path.join(project_root, DEFAULT_RAW_DIR),
        'cleansed'  : os.path.join(project_root, DEFAULT_CLEANSED_DIR),
        'results'   : os.path.join(project_root, DEFAULT_RESULTS_DIR)

    }
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
    return paths


def _loom_cross_cat(path, sample):
    """Return the loom CrossCat structure at `path`, whose id is `sample`."""
    model_in = os.path.join(
        path, 'samples', 'sample.%d' % (sample,), 'model.pb.gz')
    cross_cat = loom.schema_pb2.CrossCat()
    with open_compressed(model_in, 'rb') as f:
        cross_cat.ParseFromString(f.read())
    return cross_cat


def _retrieve_loom_store():
    """Retrieves absolute path of the loom store."""
    if os.environ.get('CGPM_LOOM_STORE'):
        return os.environ['CGPM_LOOM_STORE']
    if not os.path.exists(DEFAULT_LOOM_STORE):
        os.makedirs(DEFAULT_LOOM_STORE)
    return DEFAULT_LOOM_STORE


def _retrieve_column_partition(path, sample):
    """Return column partition from CrossCat `sample` at `path`.

    The returned structure is of the form `cgpm.crosscat.state.State.Zv`.
    """
    cross_cat = _loom_cross_cat(path, sample)
    return dict(itertools.chain.from_iterable([
        [(featureid, k) for featureid in kind.featureids]
        for k, kind in enumerate(cross_cat.kinds)
    ]))


def _retrieve_featureid_to_cgpm(path):
    """Returns a dict mapping loom's 0-based featureid to cgpm.outputs."""
    # Loom orders features alphabetically based on statistical types:
    # i.e. 'bb' < 'dd' < 'nich'. The ordering is stored in
    # `ingest/encoding.json.gz`.
    encoding_in = os.path.join(path, 'ingest', 'encoding.json.gz')
    features = json_load(encoding_in)
    def colname_to_output(cname):
        # Convert dummy column name from 'c00012' to the integer 12.
        return int(cname.replace('c', ''))
    return {
        i: colname_to_output(f['name']) for i, f in enumerate(features)
    }


def _retrieve_row_partitions(path, sample):
    """Return row partition from CrossCat `sample` at `path`.

    The returned structure is of the form `cgpm.crosscat.state.State.Zrv`.
    """
    cross_cat = _loom_cross_cat(path, sample)
    num_kinds = len(cross_cat.kinds)
    assign_in = os.path.join(
        path, 'samples', 'sample.%d' % (sample,), 'assign.pbs.gz')
    assignments = {
        a.rowid: [a.groupids(k) for k in xrange(num_kinds)]
        for a in assignment_stream_load(assign_in)
    }
    rowids = sorted(assignments)
    return {
        k: [assignments[rowid][k] for rowid in rowids]
        for k in xrange(num_kinds)
    }


def _update_state(state, path, sample):
    """Updates `state` to match the CrossCat `sample` at `path`.

    Only the row and column partitions are updated; parameter inference
    (state alpha, view alphas, hyperparameters, etc) should be transitioned
    separately.

    Wild errors will occur if the Loom object is incompatible with `state`.
    """

    # Retrieve the new column partition from loom.
    Zv_new_raw = _retrieve_column_partition(path, sample)
    assert sorted(Zv_new_raw.keys()) == range(len(state.outputs))

    # The keys of Zv are contiguous
    # from [0..len(outputs)], while state.outputs are arbitrary integers, so we
    # need to map the loom feature ids correctly.
    output_mapping = _retrieve_featureid_to_cgpm(path)
    assert sorted(output_mapping.values()) == sorted(state.outputs)
    Zv_new = {output_mapping[f]: Zv_new_raw[f] for f in Zv_new_raw}

    # Retrieve the new row partitions from loom. The view ids are contiguous
    # from [0..n_views].
    Zvr_new = _retrieve_row_partitions(path, sample)
    assert set(Zv_new.values()) == set(Zvr_new.keys())

    # Create new views in cgpm, with the corresponding loom row partitions.
    offset = max(state.views) + 1
    new_views = []
    for v in sorted(set(Zvr_new.keys())):
        index = v + offset
        assert index not in state.views
        view = View(
            state.X,
            outputs=[state.crp_id_view + index],
            Zr=Zvr_new[v],
            rng=state.rng
        )
        new_views.append(view)
        state._append_view(view, index)

    # Migrate each dim to its new view.
    for c in state.outputs:
        v_current = state.Zv(c)
        v_new = Zv_new[c] + offset
        state._migrate_dim(v_current, v_new, state.dim_for(c), reassign=True)

    assert len(state.views) == len(new_views)
    state._check_partitions()

    return state


def _update_state_mp(args):
    return _update_state(*args)


def _validate_transition(
        N=None, S=None, kernels=None, seed=None,
        checkpoint=None, progress=None):
    # These features will be implemented in stages; raise errors for now.
    if S is not None:
        raise ValueError('Loom does not support transitions by seconds.')
    if kernels is not None:
        raise ValueError('Loom does not support kernels.')
    if progress is not None:
        raise ValueError('Loom does not support progress bar.')
    if checkpoint is not None:
        raise ValueError('Loom does not support checkpoint.')


def _write_dataset(state, path):
    """Write a csv file of `state.X` to the file at `path`."""
    frame = pd.DataFrame([state.X[i] for i in state.outputs]).T
    assert frame.shape == (state.n_rows(), state.n_cols())
    frame.columns = _generate_column_names(state)
    # Update columns which can be safely converted to int.
    for col in frame.columns:
        if all(frame[col] == frame[col]//1):
            frame[col] = frame[col].astype(int)
    frame.to_csv(path, na_rep='', index=False)


def _write_schema(state, path):
    """Writes a csv file of the loom schema of `state` to the file at `path`."""
    column_names = _generate_column_names(state)
    loom_stattypes = _generate_loom_stattypes(state)
    with open(path, 'wb') as schema_file:
        writer = csv.writer(
            schema_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Feature Name', 'Type'])
        writer.writerows(zip(column_names, loom_stattypes))


# --- End of helper methods ----------------------------------------------------


def initialize(state):
    """Run the preprocessing pipeline.
    |- write dataset and schema to csv
    |- cleanse
    |- transform
    |- ingest
    |- ready for: infer
    """
    paths = _generate_project_paths()

    # Write dataset and schema csv files.
    dataset_file = os.path.join(paths['raw'], 'data.csv')
    schema_file = os.path.join(paths['raw'], 'schema.csv')

    _write_dataset(state, dataset_file)
    _write_schema(state, schema_file)

    # Cleanse and compress the dataset for Loom.
    cleansed_file = os.path.join(paths['cleansed'], 'data.csv.gz')
    loom.cleanse.force_ascii(dataset_file, cleansed_file)

    # Transform the cleansed dataset into Loom.
    loom.tasks.transform(paths['results'], schema_file, paths['cleansed'])

    # Ingest the transformed data.
    loom.tasks.ingest(paths['results'])

    return paths


def transition(
        state, N=None, S=None, kernels=None, seed=None, checkpoint=None,
        progress=None):
    """Runs full Gibbs sweeps of all kernels on the cgpm.state.State object."""

    # Check compatible transition parameters.
    _validate_transition(
        N=N, S=S, kernels=kernels, seed=seed, checkpoint=checkpoint,
        progress=progress)

    # Create Loom project if necessary.
    if state._loom_path is None:
        state._loom_path = initialize(state)

    if N is None:
        N = 1

    # The seed is used to determine which directory under results/samples
    # to use. Default to 0, since infer_one has no default mode.
    if seed is None:
        seed = 0

    loom.tasks.infer_one(
        state._loom_path['results'],
        seed=seed,
        config={"schedule": {"extra_passes": N}}
    )

    _update_state(state, state._loom_path['results'], seed)


def transition_engine(
        engine, N=None, S=None, kernels=None, seed=None, checkpoint=None,
        progress=None):
    """Run full Gibbs (all kernels) on all states in cgpm.engine.Engine object.

    Implemented separately to use Loom multiprocessing, and share a single
    Loom project among several cgpm states (Loom samples).
    """

    # Check compatible transition parameters.
    _validate_transition(
        N=N, S=S, kernels=kernels, seed=seed, checkpoint=checkpoint,
        progress=progress)

    # All the states must have the same loom project path.
    for state in engine.states:
        assert state._loom_path == engine.states[0]._loom_path

    # Create Loom project if necessary.
    if engine.states[0]._loom_path is None:
        loom_path = initialize(engine.states[0])
        for state in engine.states:
            state._loom_path = loom_path

    # Run transitions using Loom multiprocessing.
    loom.tasks.infer(
        engine.states[0]._loom_path['results'],
        sample_count=engine.num_states(),
        config={"schedule": {"extra_passes": N}}
    )

    # Update the engine and save the engine.
    args = [
        (engine.states[i], engine.states[i]._loom_path['results'], i)
        for i in xrange(engine.num_states())
    ]
    engine.states = parallel_map(_update_state_mp, args)

    # Transition the non-structural parameters.
    num_transitions = int(len(engine.states[0].outputs)**.5)
    engine.transition(
        N=num_transitions,
        kernels=['column_hypers', 'column_params', 'alpha', 'view_alphas']
    )
