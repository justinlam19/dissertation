from unittest import mock

from quantization.utils import get_attr, set_attr

class TestGetAttr:
    def test_get_single_attr(self):
        # Arrange
        mock_obj = mock.MagicMock()
        expected = "expected"
        mock_obj.attr = expected

        # Act
        attr = get_attr(mock_obj, "attr")

        # Assert
        assert attr == expected

    def test_get_nested_attr(self):
        # Arrange
        mock_obj = mock.MagicMock()
        expected = "expected"
        mock_obj.attr1.attr2.attr3 = expected

        # Act
        attr = get_attr(mock_obj, "attr1.attr2.attr3")

        # Assert
        assert attr == expected

    def test_get_element_by_index(self):
        # Arrange
        mock_list = ["not expected", "expected", "not expected 2"]
        
        # Act
        attr = get_attr(mock_list, "1")

        # Assert
        assert attr == mock_list[1]

    def test_get_nested_element_by_index(self):
        # Arrange
        mock_list = [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ]
        
        # Act
        attr = get_attr(mock_list, "1.0.2")

        # Assert
        assert attr == mock_list[1][0][2]

    def test_nested_attr_and_index(self):
        # Arrange
        expected = "expected"
        mock_obj = mock.MagicMock()
        nested_mock_obj = mock.MagicMock()
        nested_mock_obj.attr1.attr2 = expected
        mock_obj.attr.my_list = [0, False, nested_mock_obj]
        
        # Act
        attr = get_attr(mock_obj, "attr.my_list.2.attr1.attr2")

        # Assert
        assert attr == expected


class TestSetAttr:
    def test_set_single_attr(self):
        # Arrange
        mock_obj = mock.MagicMock()
        expected = "expected"

        # Act
        set_attr(mock_obj, "attr", expected)

        # Assert
        assert mock_obj.attr == expected

    def test_get_nested_attr(self):
        # Arrange
        mock_obj = mock.MagicMock()
        expected = "expected"

        # Act
        set_attr(mock_obj, "attr1.attr2.attr3", expected)

        # Assert
        assert mock_obj.attr1.attr2.attr3 == expected

    def test_get_element_by_index(self):
        # Arrange
        mock_list = [0, 1, 2]
        expected = "expected"
        
        # Act
        set_attr(mock_list, "1", expected)

        # Assert
        assert mock_list[1] == expected

    def test_get_nested_element_by_index(self):
        # Arrange
        expected = "expected"
        mock_list = [
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            [
                [7, 8, 9],
                [10, 11, 12],
            ],
        ]
        
        # Act
        set_attr(mock_list, "1.0.2", expected)

        # Assert
        assert mock_list[1][0][2] == expected

    def test_nested_attr_and_index(self):
        # Arrange
        expected = "expected"
        mock_obj = mock.MagicMock()
        nested_mock_obj = mock.MagicMock()
        mock_obj.attr.my_list = [0, False, nested_mock_obj]
        
        # Act
        set_attr(mock_obj, "attr.my_list.2.attr1.attr2", expected)

        # Assert
        assert mock_obj.attr.my_list[2].attr1.attr2 == expected   
