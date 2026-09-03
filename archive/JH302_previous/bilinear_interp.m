function val = bilinear_interp(A, x, y)
    x1 = floor(x);
    x2 = ceil(x);
    y1 = floor(y);
    y2 = ceil(y);

    if x1 < 1 || x2 > size(A,2) || y1 < 1 || y2 > size(A,1)
        val = 0;
        return;
    end

    Q11 = A(y1, x1);
    Q21 = A(y1, x2);
    Q12 = A(y2, x1);
    Q22 = A(y2, x2);

    val = Q11 * (x2 - x) * (y2 - y) + ...
          Q21 * (x - x1) * (y2 - y) + ...
          Q12 * (x2 - x) * (y - y1) + ...
          Q22 * (x - x1) * (y - y1);
end
