"""Module to preprocess main log data."""

import logging
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)


def preprocess_expert_labeling(
	path_to_labeling: Union[List[str], str],
	dataframe: pd.DataFrame,
	well_column: str,
	depth_column: str,
	formation_column: str,
	class_column: str,
	layer_column: str,
	target_column: str,
	fm_postfix: bool = False,
) -> pd.DataFrame:
	"""Preprocess excel with expert's labeling.

	:param path_to_labeling: path to excels with expert's labels
	:param dataframe: dataframe with main data
	:param well_column: name of column with names of wells
	:param depth_column: name of column with depth markup
	:param formation_column: name of column with names of formations
	:param class_column: name of column with expert's "Class" markup
	:param layer_column: name of column with expert's "Layer" markup
	:param target_column: name of column with "Target" for task (Class + Layer)
	:param fm_postfix: postfix used to identify formation (zero for NZ and Fm. for Norway)

	:return: dataframe with labels
	"""
	# delete unnecessary characters in the wells' names
	well_dataset = dataframe.copy()
	well_dataset[well_column] = well_dataset[well_column].apply(lambda x: re.sub("[^0-9A-z]", "", x))
	# support multiple labelling files
	if isinstance(path_to_labeling, str):
		path_to_labeling = [path_to_labeling]

	for filename in path_to_labeling:
		# delete unnecessary characters in the formation name (filename)
		formation = re.match("[a-zA-Z]+", filename.name)[0]

		# read labelling file
		# delete unnecessary characters in the wells' names of labelling data
		target = pd.read_excel(filename, sheet_name="Sheet1")
		if "WellName" in target:
			target = target.rename(columns={"WellName": "Well"})
		if "Top" in target:
			target = target.rename(columns={"Top": "top"})
		if "Bottom" in target:
			target = target.rename(columns={"Bottom": "bottom"})
		target["Well"] = target["Well"].apply(lambda x: re.sub("[^0-9A-z]", "", x))

		# get current and other formations data
		mask = (
			well_dataset[formation_column] == formation + " Fm."
			if fm_postfix
			else well_dataset[formation_column] == formation
		)

		data = well_dataset[mask].copy()
		other_formations = well_dataset[~mask]

		# read labelling
		for i in range(len(target)):
			data.loc[
				(data[well_column] == target["Well"].iloc[i])
				& (data[depth_column] >= target["top"].iloc[i])
				& (data[depth_column] <= target["bottom"].iloc[i]),
				class_column,
			] = target["Class"].iloc[i]
			data.loc[
				(data[well_column] == target["Well"].iloc[i])
				& (data[depth_column] >= target["top"].iloc[i])
				& (data[depth_column] <= target["bottom"].iloc[i]),
				layer_column,
			] = target["Layer"].iloc[i]

		mask = data[class_column].notna() & data[layer_column].notna()
		data[target_column] = np.nan
		data.loc[mask, target_column] = data.loc[mask, class_column].astype(str) + data.loc[mask, layer_column].astype(
			str
		)

		# encode values in target column
		label_encoder = LabelEncoder()
		data.loc[mask, target_column] = label_encoder.fit_transform(data.loc[mask, target_column])
		well_dataset = pd.concat([data, other_formations]).sort_index()

	return well_dataset


def groupby_transformation(
	data: pd.DataFrame,
	features_to_transform: List[str],
	groupby_features: Union[List[str], str],
	transformer: Callable,
	return_full: bool = False,
) -> pd.DataFrame:
	"""Auxiliary to do transformations for the selected groups.

	:param data: dataframe with main data
	:param features_to_transform: features to be transformed
	:param groupby_features: features for grouping
	:param transformer: functions (transformer) to apply to selected features
	:param return_full: if False return dataset with only transformed data ignoring others
	:return: preprocessed dataframe
	"""
	if isinstance(groupby_features, str):
		groupby_features = [groupby_features]

	# transform data through the selected groups
	transformed_data = (
		data[features_to_transform + groupby_features].groupby(groupby_features, dropna=False).transform(transformer)
	)

	# if True return all features from original dataframe dropna=False).transform(transformer)
	if return_full:
		columns_to_add = list(set(data.columns) - set(transformed_data.columns))
		transformed_data[columns_to_add] = data[columns_to_add]

	return transformed_data


def preprocessed_general_data(
	data: pd.DataFrame,
	encode_cols: Optional[List[str]] = None,
	fix_type_cols: Optional[List[Tuple[str, str]]] = None,
	fix_nan_values: Optional[List[Tuple[str, float]]] = None,
	fix_outlier_values: Optional[List[Tuple[str, float, float]]] = None,
	drop_cols: Optional[List[str]] = None,
	log_cols: Optional[List[str]] = None,
	diff_cols: Optional[List[Tuple[str, str, float]]] = None,
	norm_cols: Optional[List[str]] = None,
	group_key_cols: Optional[List[str]] = None,
	norm_group_key_cols: Optional[List[str]] = None,
	features_to_fill: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
	"""Preprocessed well log data with respect to expert's suggestions.

	:param data: dataframe with main wells' data
	:param encode_cols: columns to apply LabelEncoder to
	:param fix_type_cols: columns to transform to another type
	:param fix_nan_values: a list of tuples of (column, value) to replace value from column to NaN
	:param fix_outlier_values: a list of tuples of (column, min_value, max_value) to replace with
		NaNs all data in column outside the interval [min_value, max_value]
	:param drop_cols: columns to drop
	:param log_cols: columns to transform to log scale
	:param diff_cols: a list of tuples of (column_a, column_b, threshold) to filter data as follows:
		only keep values that have `data[column_a] - data[column_b] <= threshold
	:param norm_cols: columns to normalize
	:param group_key_cols: columns to use as groups when filling data
	:param norm_group_key_cols: columns to use as groups when normalizing data
	:param features_to_fill: features to apply backward fill and forward fill to
	:return: preprocessed dataframe and maps from encoded columns
	"""
	dataframe = data.copy()
	# fix columns types
	for column, col_type in fix_type_cols or []:
		if column in dataframe:
			LOGGER.warning("Transforming %s to %s.", column, col_type)
			dataframe[column] = dataframe[column].astype(col_type)

	# drop unnecessary columns
	if drop_cols:
		LOGGER.warning("Dropping %s.", ", ".join(drop_cols))
		dataframe = dataframe.drop(drop_cols, axis=1)

	# replace nan-outlier values with NaNs
	for column, to_replace in fix_nan_values or []:
		if column in dataframe:
			LOGGER.warning("Replacing %s in %s with NaNs.", to_replace, column)
			dataframe[column] = dataframe[column].replace(to_replace, np.nan)

	# fix common outlier values
	for column, min_value, max_value in fix_outlier_values or []:
		if column in dataframe:
			LOGGER.warning(
				"Replace values outside range [%s, %s] with NaNs in %s.",
				min_value,
				max_value,
				column,
			)
			dataframe[column] = dataframe[column].mask(~dataframe[column].between(min_value, max_value))

	# in our old preprocessing, we drop RESX columns with negative values
	# for col in log_cols or []:
	# 	LOGGER.warning("Logarithm of %s.", col)
	# 	dataframe = dataframe.drop(dataframe[col][dataframe[col] <= 0].index, axis=0)
	# 	dataframe[col] = np.log(dataframe[col].to_numpy())

	# in new version, we do it in other way, it results in quality changing for feature clustering models
	# logarithm columns
	for col in log_cols or []:
		LOGGER.warning("Logarithm of %s.", col)
		dataframe[col] = np.log(dataframe[col].to_numpy() + 1e-8)

	# filter values comparing its delta with threshold (to drop cavernous intervals)
	for first, second, threshold in diff_cols or []:
		LOGGER.warning("Leaving only %s - %s < %s.", first, second, threshold)
		dataframe.loc[:, f"delta_{first}_{second}"] = dataframe[first] - dataframe[second]
		dataframe = dataframe.drop(dataframe[dataframe[f"delta_{first}_{second}"] > threshold].index, axis=0)
		dataframe = dataframe.drop(f"delta_{first}_{second}", axis=1)

	# encode columns
	le_maps = dict()
	for column in encode_cols or []:
		LOGGER.warning("Substituting %s with ordinal codes.", column)
		label_encoder = LabelEncoder()
		dataframe[column] = label_encoder.fit_transform(dataframe[column])
		le_maps[column] = label_encoder

	# ffill and bfill features in each group
	# if there are no any observations in the well, NaNs still remains
	LOGGER.info("Features in features_to_fill are: {}".format(" ".join(map(str, features_to_fill))))

	if features_to_fill:
		if group_key_cols:
			LOGGER.info("Running <ffill>")
			dataframe = groupby_transformation(
				dataframe,
				features_to_fill,
				group_key_cols,
				lambda x: x.ffill(),
				return_full=True,
			)

			LOGGER.info("Running <bfill>")
			dataframe = groupby_transformation(
				dataframe,
				features_to_fill,
				group_key_cols,
				lambda x: x.bfill(),
				return_full=True,
			)
		else:
			LOGGER.info("Running <ffil_bfill>")
			dataframe[features_to_fill] = dataframe[features_to_fill].ffill().bfill()

	# normalized feature for each group
	if norm_group_key_cols is not None:
		LOGGER.warning("Normalizing %s using %s group.", ", ".join(norm_cols), norm_group_key_cols)

		# add 1e-8 to avoid zero division
		dataframe.loc[:, norm_cols] = groupby_transformation(
			dataframe, norm_cols, norm_group_key_cols, lambda x: (x - x.mean()) / (x.std() + 1e-8)
		)

	else:
		# normalized feature for each group
		for col in norm_cols or []:
			LOGGER.warning("Normalizing %s.", col)
			dataframe[col] = StandardScaler().fit_transform(dataframe[col].to_numpy())
	return dataframe, le_maps


def prepare_general_data_with_expert_labels(
	path_to_data: str,
	path_to_labeling: Optional[Union[List[str], str]],
	required_cols: List[str],
	well_column: str,
	depth_column: str,
	formation_column: str,
	class_column: str,
	layer_column: str,
	target_column: str,
	rename_columns: Optional[Dict[str, str]] = None,
	delimiter: str = ",",
	encode_cols: Optional[List[str]] = None,
	fix_type_cols: Optional[List[Tuple[str, str]]] = None,
	fix_nan_values: Optional[List[Tuple[str, float]]] = None,
	fix_outlier_values: Optional[List[Tuple[str, float, float]]] = None,
	drop_cols: Optional[List[str]] = None,
	log_cols: Optional[List[str]] = None,
	diff_cols: Optional[List[Tuple[str, str, float]]] = None,
	norm_cols: Optional[List[str]] = None,
	group_key_cols: Optional[List[str]] = None,
	norm_group_key_cols: Optional[List[str]] = None,
	features_to_fill: Optional[List[str]] = None,
	fm_postfix: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
	"""A very general function to read data.

	:param path_to_data: path to well logs' data in .csv format
	:param path_to_labeling: path to excel with expert's labels (very strict format)
	:param required_cols: columns to check, must be in data, depends on task
	:param well_column: name of column with names of wells
	:param depth_column: name of column with depth markup
	:param formation_column: name of column with names of formations
	:param class_column: name of column with expert's "Class" markup
	:param layer_column: name of column with expert's "Layer" markup
	:param target_column: name of column with "Target" for task (Class + Layer)
	:param rename_columns: dict from column names in data to bring to standard (New Zealand data) format
	:param delimiter: delimiter of data .csv file
	:param encode_cols: columns to apply LabelEncoder to
	:param fix_type_cols: columns to transform to another type
	:param fix_nan_values: a list of tuples of (column, value) to replace value from column to NaN
	:param fix_outlier_values: a list of tuples of (column, min_value, max_value)
		to replace with NaNs all data in column outside the interval [min_value, max_value]
	:param drop_cols: columns to drop
	:param log_cols: columns to transform to log
	:param diff_cols: a list of tuples of (column_a, column_b, threshold) to filter data as follows:
		only keep values that have `data[column_a] - data[column_b] <= threshold
	:param norm_cols: columns to normalize
	:param group_key_cols: columns to use as groups when filling data
	:param norm_group_key_cols: columns to use as groups when normalizing data
	:param features_to_fill: features to apply backward fill and forward fill to
	:param fm_postfix: postfix used to identify formation (zero for NZ and Fm. for Norway)

	:return: preprocessed dataframe and maps from encoded columns
	"""
	# read and prepare main data
	dataframe = pd.read_csv(path_to_data, low_memory=False, delimiter=delimiter)

	# bring columns to standard format
	if rename_columns is not None:
		dataframe = dataframe.rename(columns=rename_columns)

	# check required columns
	diff = set(required_cols).difference(dataframe.columns)
	if diff:
		raise ValueError(f"Missing necessary columns {diff}.")

	# check necessary columns for cavernous filtration (CALI and BS)
	if diff_cols and ("CALI" not in dataframe or "BS" not in dataframe):
		LOGGER.warning("CALI or BS features are missing, skipping this filtration.")
		diff_cols = None

	# check necessary columns for logarithm (residuals RESS, RESM, RESD)
	if log_cols and ("RESS" not in dataframe or "RESM" not in dataframe or "RESD" not in dataframe):
		LOGGER.warning("RESS, RESD or RESM features are missing, skipping this filtration.")
		log_cols = None

	# check necessary columns for normalization based on wells (GR and NEUT)
	if norm_cols and "NEUT" not in dataframe:
		LOGGER.warning("NEUT feature is missing, normalizing 'GR' only.")
		norm_cols = ["GR"]

	# add labeling to data
	if path_to_labeling is not None:
		LOGGER.warning("Adding expert labels from file.")
		dataframe = preprocess_expert_labeling(
			path_to_labeling,
			dataframe,
			well_column,
			depth_column,
			formation_column,
			class_column,
			layer_column,
			target_column,
			fm_postfix=fm_postfix,
		)
	else:
		LOGGER.warning("Expert labels are already in the data.")

	# main data preprocessing
	dataframe, le_maps = preprocessed_general_data(
		dataframe,
		encode_cols=encode_cols,
		fix_type_cols=fix_type_cols,
		fix_nan_values=fix_nan_values,
		fix_outlier_values=fix_outlier_values,
		drop_cols=drop_cols,
		log_cols=log_cols,
		diff_cols=diff_cols,
		norm_cols=norm_cols,
		group_key_cols=group_key_cols,
		norm_group_key_cols=norm_group_key_cols,
		features_to_fill=features_to_fill,
	)
	return dataframe, le_maps
