from rest_framework.pagination import CursorPagination

class MyCursorPagination(CursorPagination):
    page_size = 16
    ordering = 'created_at'