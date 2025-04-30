def predict(x, model="embc2025"):
    """
    Predicts a binary class (True/False) based on a decision tree model.

    This function implements a decision tree model that evaluates
    different signal characteristics to determine its classification.
    The function can use different models based on the model_name parameter.

    Parameters
    ----------
    x : list or numpy.ndarray
        Feature vector of dimension 14 containing the following attributes:
        - x[0]: Amplitude (multiplied by 10000 in the function)
        - x[1]: Duration (multiplied by 10000 in the function)
        - x[2]: LevelSCR (multiplied by 10000 in the function)
        - x[3]: Pente (multiplied by 10000 in the function)
        - x[4]: A_Level (multiplied by 10000 in the function)
        - x[5]: x_A_Level__Duration (multiplied by 10000 in the function)
        - x[6]: Cubic_B_x_2 (multiplied by 10000 in the function)
        - x[7]: Cubic_A_x_3 (multiplied by 10000 in the function)
        - x[8]: SlopeAfter (multiplied by 10000 in the function)
        - x[9]: Pente_Pt_Inflexion (multiplied by 10000 in the function)
        - x[10]: Cubic_A_B (multiplied by 10000 in the function)
        - x[11]: Squared_B_x (multiplied by 10000 in the function)
        - x[12]: Squared_A_x_2 (multiplied by 10000 in the function)
        - x[13]: Squared_A_B (multiplied by 10000 in the function)

    model : str, optional
        Name of the model to use for prediction. Currently supports 'embc2025' only,
        but will be expanded in the future to support additional models.
        Default is "embc2025".

    Returns
    -------
    bool
        True if the model predicts class "Y", False if the model predicts class "N"

    Raises
    ------
    ValueError
        If an unsupported model_name is provided

    Notes
    -----
    The default decision tree primarily uses A_Level and Duration features
    as the first decision points, followed by other features depending on the branches.
    All input values are multiplied by 10000 before being used in the decision logic.
    """
    # Index mapping for improved readability
    # Amplitude = x[0] * 10000
    Duration = x[1] * 10000
    LevelSCR = x[2] * 10000
    # Pente = x[3] * 10000
    A_Level = x[4] * 10000
    x_A_Level__Duration = x[5] * 10000
    # Cubic_B_x_2 = x[6] * 10000
    Cubic_A_x_3 = x[7] * 10000
    SlopeAfter = x[8] * 10000
    # Pente_Pt_Inflexion = x[9] * 10000
    # Cubic_A_B = x[10] * 10000
    # Squared_B_x = x[11] * 10000
    # Squared_A_x_2 = x[12] * 10000
    # Squared_A_B = x[13] * 10000

    # Select the appropriate model based on model_name
    if model == "embc2025":
        # Default decision tree implementation
        if A_Level < 94.5:
            if Duration < 9995:
                return False  # Class N
            else:  # Duration >= 9995
                if x_A_Level__Duration < 36.5:
                    return False  # Class N
                else:  # x_A_Level__Duration >= 36.5
                    if A_Level < 79:
                        return False  # Class N
                    else:  # A_Level >= 79
                        return True  # Class Y
        else:  # A_Level >= 94.5
            if Duration < 9995:
                if LevelSCR < 120080:
                    return False  # Class N
                else:  # LevelSCR >= 120080
                    if SlopeAfter < -1030:
                        return True  # Class Y
                    else:  # SlopeAfter >= -1030
                        return False  # Class N
            else:  # Duration >= 9995
                if Cubic_A_x_3 < -31329.5:
                    return False  # Class N
                else:  # Cubic_A_x_3 >= -31329.5
                    return True  # Class Y
    else:
        # Handle unsupported model names
        raise ValueError(f"Model '{model}' is not supported")
