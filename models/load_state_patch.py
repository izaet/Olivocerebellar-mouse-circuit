import brainpy as bp
import brainpy.helpers as helpers
from brainpy import dynsys
from brainpy.dynsys import DynamicalSystem, DynView
from typing import Dict, Callable
from brainpy.math.object_transform.base import StateLoadResult
import re


"""
Monkey patch for brainpy.helpers.load_state to fix the issue of not being able to replace/load network nodes when the names do not match.
BrainPy auto-generates unique names (PurkinjeCell2, PurkinjeCell4, …) via a global counter.

This function fixes the issue by matching the base names of the nodes, ignoring the digits attached to them.

States have the following structure, with 'PurkinjeCell13' being the brainpy.dyn.Neuron class (outer key).
"PurkinjeCell13.rho" and "PurkinjeCell13.V" are the Brainpy variables within that class (inner keys).

state['PurkinjeCell13'] == {
    "PurkinjeCell13.rho": ...,
    "PurkinjeCell13.V": ...,
    ...
}

"""
def numeric_suffix(name: str):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else -1

def base_name(name: str):
    """Strip digits from a BrainPy node name.

    e.g. 'PurkinjeCell2' -> 'PurkinjeCell'
    """
    return re.sub(r'\d+$', '', name) # Substitute suffix with empty string

def node_signature_variables(node_state: Dict):
    """
    Obtain a signature of the node's state dictionary, including the variable names, their shapes/dtypes and the order of variables."""
    if not isinstance(node_state, dict):
        return None
    
    items = []
    for key, value in sorted(node_state.items()):
        suffix = key.split(".", 1)[-1]
        if hasattr(value, "shape"):
            items.append((suffix, tuple(value.shape), str(getattr(value, "dtype", None))))
        else:
            items.append((suffix, type(value).__name__))
    return tuple(items)


def remap_inner_keys(old_name: str, new_name: str, node_state: Dict):
    """
    Renaming the inner keys of a node's state dictionary to match the new node name.

    args:
        old_name: The original name of the node (e.g., 'PurkinjeCell13').
        new_name: The new name of the node (e.g., 'PurkinjeCell2').
        node_state: The state dictionary of the node, where keys are in the format 'old_name.variable_name'.    

    returns:
        A new state dictionary with keys renamed to use the new node name.
    """

    if not isinstance(node_state, dict):
        return node_state

    remapped = {}
    old_prefix = old_name + "."
    new_prefix = new_name + "."
    for key, value in node_state.items():
            if isinstance(key, str) and key.startswith(old_prefix):
                remapped[new_prefix + key[len(old_prefix):]] = value
            else:
                remapped[key] = value

    return remapped

def load_state_fixed(target: DynamicalSystem, state_dict: Dict, **kwargs):
    """
    Load the state of a DynamicalSystem, remapping node names to match the current network digit suffixes. 
    The state dictionary values are loaded into the corresponding nodes of the target network.

    Args:
        target: The DynamicalSystem to load the state into.
        state_dict: A dictionary containing the state to load, with keys as node names and values as their states.
    returns:
        A StateLoadResult object containing lists of missing and unexpected keys. 
    
    """

    if not isinstance(state_dict, dict):
        return helpers.load_state(target, state_dict, **kwargs)

    # Clear current network state and inputs 
    try:
        helpers.reset_state(target)
        helpers.clear_input(target)
    except Exception:
        pass
   
    # Map node names in state_dict to their base names
    node_by_base = {}
    node_signatures = {}

    for key, variables in state_dict.items():
        if not isinstance(key, str):
            continue
        base = base_name(key)
        node_by_base.setdefault(base, []).append(key)
        node_signatures[key] = node_signature_variables(variables)

    nodes = target.nodes().subset(DynamicalSystem).not_subset(DynView).unique()
    missing_keys = []
    unexpected_keys = []
    
    # Remap outer keys
    for name, node in nodes.items():
        key_to_use = None
        old_name = None

        
        # Exact name match
        if name in state_dict:
            key_to_use = name
            old_name = name

        # Base-name match
        else:
            candidates = node_by_base.get(base_name(name), [])
            if len(candidates) == 1:
                key_to_use = candidates[0]
                old_name = candidates[0]

            # When there are multiple node-base matches
            elif len(candidates) > 1:

                 # Compare node class/types aka signatures
                target_sig = node_signature_variables(node.save_state())
                matches = [cand for cand in candidates if node_signatures.get(cand) == target_sig]

                if len(matches) == 1:
                    key_to_use = matches[0]
                    old_name = matches[0]

                # Fallback to matching by ordering of nodes 
                else: 
                    sorted_candidates = sorted(candidates, key=numeric_suffix)
                    same_base_nodes = [n for n in nodes if base_name(n) == base_name(name)]    

                    if len(sorted_candidates) == len(same_base_nodes):
                        idx = sorted(same_base_nodes, key= numeric_suffix).index(name)

                        key_to_use = sorted_candidates[idx]
                        old_name = key_to_use


            if key_to_use is None:
                missing_keys.append(name)
                continue
        
        # Remap inner keys to new name
        node_state_raw = state_dict[key_to_use]
        node_state = remap_inner_keys(old_name, name, node_state_raw)

        # Load the state into the node
        r= node.load_state(node_state, **kwargs)
        if r is not None:
            missing, unexpected = r
            missing_keys.extend([f'{name}.{key}' for key in missing])
            unexpected_keys.extend([f'{name}.{key}' for key in unexpected])

    return StateLoadResult(missing_keys, unexpected_keys)

helpers.load_state = load_state_fixed
bp.load_state = load_state_fixed  

    