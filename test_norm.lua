require("torch")
local ts = torch.Tensor(3,5)
for i = 1, 3 do
    for j = 1, 5 do
        ts[i][j] = (i-1)*5+j-1

    end
end
print(ts)
print(torch.norm(ts,2,2):expandAs(ts))

print(ts:cdiv(torch.norm(ts, 2, 2):expandAs(ts)))