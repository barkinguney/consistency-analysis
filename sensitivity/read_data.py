"""Disclaimer: Created this script mostly with LLMs. Might not cover all cases."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import xml.etree.ElementTree as ET
import cantera as ct


Number = Union[int, float]


def _strip_ns(tag: str) -> str:
    """Remove XML namespace if present: {ns}tag -> tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _as_number(s: Optional[str]) -> Optional[Number]:
    """Parse int/float from text; return None on missing/empty."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    # try int first for cleaner values
    try:
        i = int(s)
        return i
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return None


def _child_text(el: ET.Element, child_tag: str) -> Optional[str]:
    """Get direct child text (ignores namespaces)."""
    for ch in list(el):
        if _strip_ns(ch.tag) == child_tag:
            return ch.text
    return None


def _find_first(root: ET.Element, tag: str) -> Optional[ET.Element]:
    """Find first element by localname tag anywhere under root."""
    for el in root.iter():
        if _strip_ns(el.tag) == tag:
            return el
    return None


def _find_all(parent: ET.Element, tag: str) -> List[ET.Element]:
    """Find all elements by localname tag anywhere under parent."""
    out: List[ET.Element] = []
    for el in parent.iter():
        if _strip_ns(el.tag) == tag:
            out.append(el)
    return out


@dataclass
class IDTPropertyDef:
    prop_id: str
    name: str
    label: Optional[str]
    units: Optional[str]
    sourcetype: Optional[str]


@dataclass
class IDTDataGroup:
    group_id: str
    # property definitions keyed by x-id (e.g., x1, x2, x3)
    properties: Dict[str, IDTPropertyDef]
    # rows as dicts keyed by x-id AND also by label/name for convenience
    rows: List[Dict[str, Any]]


@dataclass
class IDTExperiment:
    file_author: Optional[str]
    file_doi: Optional[str]
    experiment_type: Optional[str]
    apparatus_kind: Optional[str]
    apparatus_mode: Optional[str]
    initial_composition: List[Dict[str, Any]]
    data_groups: List[IDTDataGroup]
    ignition_type: Optional[IgnitionType]
    common_properties: Dict[str, Any]
    common_property_units: Dict[str, Optional[str]]
    
@dataclass
class IgnitionType:
    target: Optional[str]
    type: Optional[str]
    amount: Optional[str]
    units: Optional[str]


def parse_idt_xml(path: Union[str, Path]) -> IDTExperiment:
    """
    Parse an IDT (ignition delay time) XML file in the shown ReSpecTh-like format.

    Returns a structured IDTExperiment containing metadata, composition, and all dataGroups.
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()

    # If the file uses namespaces, ElementTree tags will be {ns}experiment etc.
    # We search by localname to stay robust.

    exp = root if _strip_ns(root.tag) == "experiment" else _find_first(root, "experiment")
    if exp is None:
        raise ValueError("Could not find <experiment> root element")

    file_author = _child_text(exp, "fileAuthor")
    file_doi = _child_text(exp, "fileDOI")
    experiment_type = _child_text(exp, "experimentType")

    apparatus = _find_first(exp, "apparatus")
    apparatus_kind = _child_text(apparatus, "kind") if apparatus is not None else None
    apparatus_mode = _child_text(apparatus, "mode") if apparatus is not None else None

    # Parse initial composition (commonProperties/property[@name="initial composition"]/component...)
    initial_composition: List[Dict[str, Any]] = []
    common_props = _find_first(exp, "commonProperties")
    if common_props is not None:
        for prop in list(common_props):
            if _strip_ns(prop.tag) != "property":
                continue
            if prop.attrib.get("name", "").strip().lower() != "initial composition":
                continue

            for comp in list(prop):
                if _strip_ns(comp.tag) != "component":
                    continue

                species_link = None
                amount_el = None
                for ch in list(comp):
                    t = _strip_ns(ch.tag)
                    if t == "speciesLink":
                        species_link = ch
                    elif t == "amount":
                        amount_el = ch

                species: Dict[str, Any] = {}
                if species_link is not None:
                    species = {
                        "preferredKey": species_link.attrib.get("preferredKey"),
                        "chemName": species_link.attrib.get("chemName"),
                        "CAS": species_link.attrib.get("CAS"),
                        "InChI": species_link.attrib.get("InChI"),
                        "SMILES": species_link.attrib.get("SMILES"),
                    }

                amount: Dict[str, Any] = {}
                if amount_el is not None:
                    amount = {
                        "units": amount_el.attrib.get("units"),
                        "value": _as_number(amount_el.text),
                    }

                if species or amount:
                    initial_composition.append({**species, **amount})

            break  # only one initial composition property expected
    
    # Parse ignition type
    ign_el = _find_first(exp, "ignitionType")
    ignition_type = None
    if ign_el is not None:
        ignition_type = IgnitionType(
            target=ign_el.attrib.get("target"),
            type=ign_el.attrib.get("type"),
            amount=ign_el.attrib.get("amount"),  # Ensure amount is included
            units=ign_el.attrib.get("units")     # Ensure units is included
        )

    # Parse data groups
    data_groups: List[IDTDataGroup] = []
    for dg in _find_all(exp, "dataGroup"):
        group_id = dg.attrib.get("id", "")

        # property definitions are direct children <property .../>
        props: Dict[str, IDTPropertyDef] = {}
        for ch in list(dg):
            if _strip_ns(ch.tag) != "property":
                continue
            pid = ch.attrib.get("id")
            if not pid:
                continue
            props[pid] = IDTPropertyDef(
                prop_id=pid,
                name=ch.attrib.get("name", ""),
                label=ch.attrib.get("label"),
                units=ch.attrib.get("units"),
                sourcetype=ch.attrib.get("sourcetype"),
            )

        # datapoints: each <dataPoint><x1>..</x1><x2>..</x2>..</dataPoint>
        rows: List[Dict[str, Any]] = []
        for dp in list(dg):
            if _strip_ns(dp.tag) != "dataPoint":
                continue

            row: Dict[str, Any] = {}
            for val_el in list(dp):
                xid = _strip_ns(val_el.tag)  # x1, x2, x3...
                row[xid] = _as_number(val_el.text)

                # also add convenience keys by label and by name if available
                if xid in props:
                    pdef = props[xid]
                    if pdef.label:
                        row[pdef.label] = row[xid]
                    if pdef.name:
                        row[pdef.name] = row[xid]

            rows.append(row)

        data_groups.append(IDTDataGroup(group_id=group_id, properties=props, rows=rows))

    # Parse common properties (temperature, pressure outside dataGroups)
    common_properties: Dict[str, Any] = {}
    common_property_units: Dict[str, Optional[str]] = {}
    common_props = _find_first(exp, "commonProperties")
    if common_props is not None:
        for prop in list(common_props):
            if _strip_ns(prop.tag) != "property":
                continue
            prop_name = prop.attrib.get("name", "").strip().lower()
            if prop_name in ["temperature", "pressure"]:
                value_el = _find_first(prop, "value")
                if value_el is not None and value_el.text:
                    common_properties[prop_name] = _as_number(value_el.text)
                    common_property_units[prop_name] = prop.attrib.get("units") or value_el.attrib.get("units") if hasattr(value_el, "attrib") else None
    
    return IDTExperiment(
        file_author=file_author,
        file_doi=file_doi,
        experiment_type=experiment_type,
        apparatus_kind=apparatus_kind,
        apparatus_mode=apparatus_mode,
        initial_composition=initial_composition,
        data_groups=data_groups,
        ignition_type=ignition_type,
        common_properties=common_properties,
        common_property_units=common_property_units,
    )


def idt_to_dataframe(exp: IDTExperiment, group_id: Optional[str] = None):
    """
    Optional helper: convert a dataGroup to a pandas DataFrame.
    Requires pandas installed.
    """
    import pandas as pd  # type: ignore

    groups = exp.data_groups
    if group_id is not None:
        groups = [g for g in groups if g.group_id == group_id]
        if not groups:
            raise ValueError(f"No dataGroup found with id={group_id!r}")

    # Use first group by default
    g = groups[0]
    df = pd.DataFrame(g.rows)

    # Prefer columns in x-id order + (label) if present
    xids = sorted(g.properties.keys(), key=lambda s: int(s[1:]) if s[1:].isdigit() else s)
    preferred_cols: List[str] = []
    for xid in xids:
        preferred_cols.append(xid)
        lab = g.properties[xid].label
        if lab and lab in df.columns:
            preferred_cols.append(lab)

    # keep preferred first, then the rest
    remaining = [c for c in df.columns if c not in preferred_cols]
    df = df[preferred_cols + remaining]
    return df


def format_composition(initial_composition: List[Dict[str, Any]]) -> str:
    """
    Format initial composition as a string like 'C2H4:0.01, O2:0.03, AR:0.96'.
    Uses preferredKey as species identifier and value as the mole fraction.
    """
    parts = []
    for comp in initial_composition:
        species = comp.get("preferredKey", "")
        value = comp.get("value")
        if species and value is not None:
            parts.append(f"{species}:{value}")
    return ", ".join(parts)


def calculate_phi_cantera(
    composition: Union[str, List[Dict[str, Any]]],
    mechanism: str = "gri30.yaml",
    fuel: Optional[str] = None,  # kept for signature compatibility
    oxidizer: str = "O2:1.0",    # kept for signature compatibility
) -> Optional[float]:
    """
    Calculate equivalence ratio (phi) by balancing required O2 against available O2.
    Diluent species (e.g., Ar) do not affect the result.
    """
    try:
        gas = ct.Solution(mechanism)
        name_map = {sp.upper(): sp for sp in gas.species_names}

        # Parse composition to dict
        if isinstance(composition, str):
            comp_dict: Dict[str, float] = {}
            for part in composition.split(","):
                part = part.strip()
                if ":" in part:
                    sp, val = part.split(":")
                    try:
                        comp_dict[sp.strip()] = float(val.strip())
                    except ValueError:
                        continue
        else:
            comp_dict = {
                comp.get("preferredKey", ""): float(comp.get("value", 0))
                for comp in composition
                if comp.get("preferredKey") and comp.get("value") is not None
            }

        # Normalize species names to mechanism names
        norm_comp: Dict[str, float] = {}
        for sp, val in comp_dict.items():
            mech_sp = name_map.get(sp.upper())
            if mech_sp and val > 0:
                norm_comp[mech_sp] = val

        if not norm_comp:
            return None

        o2_available = norm_comp.get("O2", 0.0)
        if o2_available <= 0:
            return None

        # Total O2 needed to fully oxidize all fuel species
        total_o2_needed = 0.0
        for sp, moles in norm_comp.items():
            if sp == "O2" or moles <= 0:
                continue
            sp_obj = gas.species(sp)
            c = sp_obj.composition.get("C", 0.0)
            h = sp_obj.composition.get("H", 0.0)
            o = sp_obj.composition.get("O", 0.0)
            nu_o2 = c + h / 4.0 - o / 2.0
            if nu_o2 > 0:
                total_o2_needed += nu_o2 * moles

        if total_o2_needed <= 0:
            return None

        return total_o2_needed / o2_available
    except Exception as e:
        print(f"Error calculating phi with Cantera: {e}")
        return None


def extract_idt_data_to_dataframe(folder_path: Union[str, Path], mechanism: str = "gri30.yaml") -> "pd.DataFrame":
    """
    Extract IDT data from all XML files in a folder into a pandas DataFrame.
    
    Args:
        folder_path: Path to folder containing XML files
        mechanism: Cantera mechanism file for phi calculation (default: gri30.yaml)
    
    Returns a DataFrame with columns: T5, P5, composition, phi, tau, ignition_type, filename
    where T5 is temperature, P5 is pressure, tau is ignition delay time, and ignition_type
    is formatted as 'target;type'.
    """
    import pandas as pd  # type: ignore
    
    folder_path = Path(folder_path)
    all_data = []
    
    for file_path in folder_path.glob("*.xml"):
        try:
            exp = parse_idt_xml(file_path)
            composition_str = format_composition(exp.initial_composition)
            
            # Format ignition type
            ignition_type_str = ""
            ignition_amount = ""
            ignition_units = ""
            if exp.ignition_type:
                target = exp.ignition_type.target or ""              
                type_val = exp.ignition_type.type or ""
                ignition_amount = exp.ignition_type.amount if exp.ignition_type.amount is not None else ""
                ignition_units = exp.ignition_type.units if exp.ignition_type.units is not None else ""
                ignition_type_str = f"{target}{type_val}"
            
            # Calculate phi using Cantera
            phi = calculate_phi_cantera(exp.initial_composition, mechanism=mechanism)
            
            # Extract units from data group properties
            T_units = ""
            P_units = ""
            Tau_units = ""
            for group in exp.data_groups:
                for prop_id, prop_def in group.properties.items():
                    if prop_def.name.lower() == "temperature" or (prop_def.label and prop_def.label.lower() == "t"):
                        T_units = prop_def.units or T_units
                    elif prop_def.name.lower() == "pressure" or (prop_def.label and prop_def.label.lower() == "p"):
                        P_units = prop_def.units or P_units
                    elif prop_def.name.lower() == "ignition delay" or (prop_def.label and prop_def.label.lower() == "tau"):
                        Tau_units = prop_def.units or Tau_units
            if not T_units:
                T_units = exp.common_property_units.get("temperature") or ""
            if not P_units:
                P_units = exp.common_property_units.get("pressure") or ""
            # Extract data from all data groups
            for group in exp.data_groups:
                for row in group.rows:
                    # Look for temperature (T), pressure (P), and ignition delay (tau)
                    # Use common properties if not in row, otherwise use row values
                    T = row.get("T") or row.get("temperature") or row.get("Temperature") or exp.common_properties.get("temperature")
                    P = row.get("P") or row.get("pressure") or row.get("Pressure") or exp.common_properties.get("pressure")
                    Tau = row.get("tau") or row.get("ignition delay") or row.get("Ignition Delay")
                    
                    all_data.append({
                        "T5": T,
                        "T5_units": T_units,
                        "P5": P,
                        "P5_units": P_units,
                        "composition": composition_str,
                        "phi": phi,
                        "tau": Tau,
                        "tau_units": Tau_units,
                        "ignition_type": ignition_type_str,
                        "ignition_target": exp.ignition_type.target.upper(),
                        "ignition_type": exp.ignition_type.type,
                        "ignition_amount": ignition_amount,
                        "ignition_units": ignition_units,
                        "filename": file_path.name
                    })
        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")
    
    return pd.DataFrame(all_data)


if __name__ == "__main__":
    # Example usage
    folder_path = "data/idt_data/syngas"
    
    # Create DataFrame with all IDT data
    df = extract_idt_data_to_dataframe(folder_path)
    df.to_csv("idt_data_summary.csv", index=False)
    print("\nIDT Data Summary:")
    print(df.head(10))
    print(f"\nTotal data points: {len(df)}")




