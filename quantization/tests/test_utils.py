from unittest.mock import MagicMock

from quantization.utils import get_module, set_module


class TestGetAttr:
    def test_get_single_attr(self):
        # GIVEN
        #      a single attribute with no nesting is specified
        mock_model = MagicMock()
        expected = "expected"
        mock_model.mods.attr = expected

        # WHEN
        #      the attribute is retrieved from the model
        attr = get_module(mock_model, "attr")

        # THEN
        #      the attribute is correctly retrieved
        assert attr == expected

    def test_get_nested_attr(self):
        # GIVEN
        #      a nested attribute is specified
        mock_model = MagicMock()
        expected = "expected"
        mock_model.mods.attr1.attr2.attr3 = expected

        # WHEN
        #      the attribute is retrieved from the model
        attr = get_module(mock_model, "attr1.attr2.attr3")

        # THEN
        #      the attribute is correctly retrieved
        assert attr == expected

    def test_get_element_by_index(self):
        # GIVEN
        #      the model contains a list
        #      an attribute inside the list is specified
        mock_model = MagicMock()
        expected = "expected"
        mock_model.mods.mock_list = [0, expected, 2]

        # WHEN
        #      the attribute is retrieved by list index
        attr = get_module(mock_model, "mock_list.1")

        # THEN
        #      the attribute is correctly retrieved
        assert attr == expected

    def test_get_nested_element_by_index(self):
        # GIVEN
        #      an attribute is inside a nested list
        #      the list is an attribute of a model
        mock_model = MagicMock()
        expected = "expected"
        mock_model.mods.mock_list = [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, expected],
                [10, 11, 12],
            ],
        ]

        # WHEN
        #      the attribute is retrieved from the model
        attr = get_module(mock_model, "mock_list.1.0.2")

        # THEN
        #      the attribute is correctly retrieved
        assert attr == expected

    def test_get_nested_attr_and_index(self):
        # GIVEN
        #      a submodule is inside a list
        #      an attribute of the submodule is specified
        expected = "expected"
        mock_model = MagicMock()
        nested_mock_module = MagicMock()
        nested_mock_module.attr1.attr2 = expected
        mock_model.mods.attr.my_list = [0, False, nested_mock_module]

        # WHEN
        #      the attribute is retrieved from the model's submodule
        attr = get_module(mock_model, "attr.my_list.2.attr1.attr2")

        # THEN
        #      the attribute is correctly retrieved
        assert attr == expected


class TestSetAttr:
    def test_set_single_attr(self):
        # GIVEN
        #      a single attribute with no nesting is specified
        mock_model = MagicMock()
        attr = "attr"
        expected = "expected"

        # WHEN
        #      the attribute is set to a value
        set_module(mock_model, attr, expected)

        # THEN
        #      the attribute is correctly set
        assert mock_model.mods.attr == expected

    def test_set_nested_attr(self):
        # GIVEN
        #      a nested attribute is specified
        mock_model = MagicMock()
        attr = "attr1.attr2.attr3"
        expected = "expected"

        # WHEN
        #      the attribue is set to a value
        set_module(mock_model, attr, expected)

        # THEN
        #      the attribute is correctly set
        assert mock_model.mods.attr1.attr2.attr3 == expected

    def test_set_element_by_index(self):
        # GIVEN
        #      an attribute is specified to be a list element
        mock_model = MagicMock()
        expected = "expected"
        mock_model.mods.mock_list = [0, 1, 2]
        attr = "mock_list.1"

        # WHEN
        #      the attribute is set to a value
        set_module(mock_model, attr, expected)

        # THEN
        #      the attribute is correctly set
        assert mock_model.mods.mock_list[1] == expected

    def test_set_nested_element_by_index(self):
        # GIVEN
        #      an attribute is specified to be a nested element inside a list
        mock_model = MagicMock()
        expected = "expected"
        mock_model.mods.mock_list = [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ]
        attr = "mock_list.1.0.2"

        # WHEN
        #      the attribute is set to a value
        set_module(mock_model, attr, expected)

        # THEN
        #      the attribute is correctly set
        assert mock_model.mods.mock_list[1][0][2] == expected

    def test_set_nested_attr_and_index(self):
        # GIVEN
        #      an attribute of a submodule is specified
        #      the submodule is an element of a list inside the model
        expected = "expected"
        mock_model = MagicMock()
        nested_mock_module = MagicMock()
        mock_model.mods.attr.my_list = [0, False, nested_mock_module]
        attr = "attr.my_list.2.attr1.attr2"

        # WHEN
        #      the attribute is set to a value
        set_module(mock_model, attr, expected)

        # THEN
        #      the attribute is correctly set
        assert mock_model.mods.attr.my_list[2].attr1.attr2 == expected
